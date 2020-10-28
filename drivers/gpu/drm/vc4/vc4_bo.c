// SPDX-License-Identifier: GPL-2.0-only
/*
 *  Copyright © 2015 Broadcom
 */

/**
 * DOC: VC4 GEM BO management support
 *
 * The VC4 GPU architecture (both scanout and rendering) has direct
 * access to system memory with no MMU in between.  To support it, we
 * use the GEM CMA helper functions to allocate contiguous ranges of
 * physical memory for our BOs.
 *
 * Since the CMA allocator is very slow, we keep a cache of recently
 * freed BOs around so that the kernel's allocation of objects for 3D
 * rendering can return quickly.
 */

#include <linux/dma-buf.h>
#include <linux/dmaengine.h>
#include <drm/drm_fb_helper.h>

#include "vc4_drv.h"
#include "uapi/drm/vc4_drm.h"

/* Limit the amount of CMA memory allocated to 128MB */
#define VC4_CMA_POOL_SIZE (128 * 1024 * 1024)

static const char * const bo_type_names[] = {
	"kernel",
	"V3D",
	"V3D shader",
	"dumb",
	"binner",
	"RCL",
	"BCL",
	"kernel BO cache",
};

static bool is_user_label(int label)
{
	return label >= VC4_BO_TYPE_COUNT;
}

static void vc4_bo_stats_print(struct drm_printer *p, struct vc4_dev *vc4)
{
	int i;

	for (i = 0; i < vc4->num_labels; i++) {
		if (!vc4->bo_labels[i].num_allocated)
			continue;

		drm_printf(p, "%30s: %6dkb BOs (%d)\n",
			   vc4->bo_labels[i].name,
			   vc4->bo_labels[i].size_allocated / 1024,
			   vc4->bo_labels[i].num_allocated);
	}

	mutex_lock(&vc4->purgeable.lock);
	if (vc4->purgeable.num)
		drm_printf(p, "%30s: %6zdkb BOs (%d)\n", "userspace BO cache",
			   vc4->purgeable.size / 1024, vc4->purgeable.num);

	if (vc4->purgeable.purged_num)
		drm_printf(p, "%30s: %6zdkb BOs (%d)\n", "total purged BO",
			   vc4->purgeable.purged_size / 1024,
			   vc4->purgeable.purged_num);
	mutex_unlock(&vc4->purgeable.lock);
}

static int vc4_bo_stats_debugfs(struct seq_file *m, void *unused)
{
	struct drm_info_node *node = (struct drm_info_node *)m->private;
	struct drm_device *dev = node->minor->dev;
	struct vc4_dev *vc4 = to_vc4_dev(dev);
	struct drm_printer p = drm_seq_file_printer(m);

	vc4_bo_stats_print(&p, vc4);

	return 0;
}

/* Takes ownership of *name and returns the appropriate slot for it in
 * the bo_labels[] array, extending it as necessary.
 *
 * This is inefficient and could use a hash table instead of walking
 * an array and strcmp()ing.  However, the assumption is that user
 * labeling will be infrequent (scanout buffers and other long-lived
 * objects, or debug driver builds), so we can live with it for now.
 */
static int vc4_get_user_label(struct vc4_dev *vc4, const char *name)
{
	int i;
	int free_slot = -1;

	for (i = 0; i < vc4->num_labels; i++) {
		if (!vc4->bo_labels[i].name) {
			free_slot = i;
		} else if (strcmp(vc4->bo_labels[i].name, name) == 0) {
			kfree(name);
			return i;
		}
	}

	if (free_slot != -1) {
		WARN_ON(vc4->bo_labels[free_slot].num_allocated != 0);
		vc4->bo_labels[free_slot].name = name;
		return free_slot;
	} else {
		u32 new_label_count = vc4->num_labels + 1;
		struct vc4_label *new_labels =
			krealloc(vc4->bo_labels,
				 new_label_count * sizeof(*new_labels),
				 GFP_KERNEL);

		if (!new_labels) {
			kfree(name);
			return -1;
		}

		free_slot = vc4->num_labels;
		vc4->bo_labels = new_labels;
		vc4->num_labels = new_label_count;

		vc4->bo_labels[free_slot].name = name;
		vc4->bo_labels[free_slot].num_allocated = 0;
		vc4->bo_labels[free_slot].size_allocated = 0;

		return free_slot;
	}
}

static void vc4_bo_set_label(struct drm_gem_object *gem_obj, int label)
{
	struct vc4_bo *bo = to_vc4_bo(gem_obj);
	struct vc4_dev *vc4 = to_vc4_dev(gem_obj->dev);

	lockdep_assert_held(&vc4->bo_lock);

	if (label != -1) {
		vc4->bo_labels[label].num_allocated++;
		vc4->bo_labels[label].size_allocated += gem_obj->size;
	}

	vc4->bo_labels[bo->label].num_allocated--;
	vc4->bo_labels[bo->label].size_allocated -= gem_obj->size;

	if (vc4->bo_labels[bo->label].num_allocated == 0 &&
	    is_user_label(bo->label)) {
		/* Free user BO label slots on last unreference.
		 * Slots are just where we track the stats for a given
		 * name, and once a name is unused we can reuse that
		 * slot.
		 */
		kfree(vc4->bo_labels[bo->label].name);
		vc4->bo_labels[bo->label].name = NULL;
	}

	bo->label = label;
}

static void vc4_bo_remove_from_pool(struct vc4_bo *bo)
{
	if (bo->buffer_copy) {
		vfree(bo->buffer_copy);
	} else {
		list_del(&bo->mru_buffers_head);
		list_del(&bo->offset_node.head);
	}
}

static void finish_cma_pool_dma_memcpy(struct vc4_bo *bo)
{
	struct vc4_dev *vc4 = to_vc4_dev(bo->base.dev);
	int ret;

	if (bo->cma_pool_dma_cookie <= 0)
		return;

	ret = dma_sync_wait(vc4->cma_pool.dma_chan, bo->cma_pool_dma_cookie);
	if (ret)
		DRM_ERROR("Failed to wait for DMA: %d\n", ret);

	bo->cma_pool_dma_cookie = 0;
	list_del(&bo->stub_offset_node.head);
}

static void vc4_bo_destroy(struct vc4_bo *bo)
{
	struct drm_gem_object *obj = &bo->base;
	struct vc4_dev *vc4 = to_vc4_dev(obj->dev);

	if (bo->validated_shader) {
		kfree(bo->validated_shader->uniform_addr_offsets);
		kfree(bo->validated_shader->texture_samples);
		kfree(bo->validated_shader);
		bo->validated_shader = NULL;
	}

	mutex_lock(&vc4->bo_lock);

	finish_cma_pool_dma_memcpy(bo);

	vc4_bo_set_label(obj, -1);

	if (bo->madv != __VC4_MADV_PURGED)
		vc4_bo_remove_from_pool(bo);

	mutex_unlock(&vc4->bo_lock);

	drm_gem_object_release(obj);

	printk("@@@ destroy %p\n", bo);

	kfree(bo);
}

void vc4_bo_add_to_purgeable_pool(struct vc4_bo *bo)
{
	struct vc4_dev *vc4 = to_vc4_dev(bo->base.dev);

	mutex_lock(&vc4->purgeable.lock);
	list_add_tail(&bo->purgeable_head, &vc4->purgeable.list);
	vc4->purgeable.num++;
	vc4->purgeable.size += bo->base.size;
	mutex_unlock(&vc4->purgeable.lock);
}

static void vc4_bo_remove_from_purgeable_pool_locked(struct vc4_bo *bo)
{
	struct vc4_dev *vc4 = to_vc4_dev(bo->base.dev);

	/* list_del_init() is used here because the caller might release
	 * the purgeable lock in order to acquire the madv one and update the
	 * madv status.
	 * During this short period of time a user might decide to mark
	 * the BO as unpurgeable, and if bo->madv is set to
	 * VC4_MADV_DONTNEED it will try to remove the BO from the
	 * purgeable list which will fail if the ->next/prev fields
	 * are set to LIST_POISON1/LIST_POISON2 (which is what
	 * list_del() does).
	 * Re-initializing the list element guarantees that list_del()
	 * will work correctly even if it's a NOP.
	 */
	list_del_init(&bo->purgeable_head);
	vc4->purgeable.num--;
	vc4->purgeable.size -= bo->base.size;
}

void vc4_bo_remove_from_purgeable_pool(struct vc4_bo *bo)
{
	struct vc4_dev *vc4 = to_vc4_dev(bo->base.dev);

	mutex_lock(&vc4->purgeable.lock);
	vc4_bo_remove_from_purgeable_pool_locked(bo);
	mutex_unlock(&vc4->purgeable.lock);
}

static void vc4_bo_purge(struct drm_gem_object *obj)
{
	struct vc4_bo *bo = to_vc4_bo(obj);
	struct drm_device *dev = obj->dev;

	WARN_ON(!mutex_is_locked(&bo->madv_lock));
	WARN_ON(bo->madv != VC4_MADV_DONTNEED);

	lockdep_assert_held(&to_vc4_dev(dev)->bo_lock);

	drm_vma_node_unmap(&obj->vma_node, dev->anon_inode->i_mapping);

	bo->madv = __VC4_MADV_PURGED;

	vc4_bo_remove_from_pool(bo);
}

static void vc4_bo_userspace_cache_purge(struct vc4_dev *vc4)
{
	mutex_lock(&vc4->purgeable.lock);
	while (!list_empty(&vc4->purgeable.list)) {
		struct vc4_bo *bo = list_first_entry(&vc4->purgeable.list,
						     struct vc4_bo,
						     purgeable_head);
		struct drm_gem_object *obj = &bo->base;
		size_t purged_size = 0;

		vc4_bo_remove_from_purgeable_pool_locked(bo);

		/* Release the purgeable lock while we're purging the BO so
		 * that other people can continue inserting things in the
		 * purgeable pool without having to wait for all BOs to be
		 * purged.
		 */
		mutex_unlock(&vc4->purgeable.lock);
		mutex_lock(&bo->madv_lock);

		/* Since we released the purgeable pool lock before acquiring
		 * the BO madv one, the user may have marked the BO as WILLNEED
		 * and re-used it in the meantime.
		 * Before purging the BO we need to make sure
		 * - it is still marked as DONTNEED
		 * - it has not been re-inserted in the purgeable list
		 * - it is not used by HW blocks
		 * If one of these conditions is not met, just skip the entry.
		 */
		if (bo->madv == VC4_MADV_DONTNEED &&
		    list_empty(&bo->purgeable_head) &&
		    !refcount_read(&bo->usecnt)) {
			purged_size = bo->base.size;
			vc4_bo_purge(obj);
		}
		mutex_unlock(&bo->madv_lock);
		mutex_lock(&vc4->purgeable.lock);

		if (purged_size) {
			vc4->purgeable.purged_size += purged_size;
			vc4->purgeable.purged_num++;
		}
	}
	mutex_unlock(&vc4->purgeable.lock);
}

static bool check_finished_cma_stub_node(struct vc4_offset_node *node)
{
	struct vc4_dev *vc4;
	struct vc4_bo *bo;
	enum dma_status status;

	if (node->type != VC4_OFFSET_NODE_TYPE_STUB)
		return false;

	bo = container_of(node, struct vc4_bo, stub_offset_node);
	vc4 = to_vc4_dev(bo->base.dev);
	status = dma_async_is_tx_complete(vc4->cma_pool.dma_chan,
					  bo->cma_pool_dma_cookie,
					  NULL, /* last */
					  NULL /* used */);

	switch (status) {
	case DMA_ERROR:
		DRM_ERROR("DMA transfer for CMA pool compaction failed\n");
		/* flow through */
	case DMA_COMPLETE:
		bo->cma_pool_dma_cookie = 0;
		list_del(&bo->stub_offset_node.head);
		return true;
	case DMA_IN_PROGRESS:
	case DMA_PAUSED:
		break;
	}

	return false;
}

static bool page_out_buffer(struct vc4_bo *buf)
{
	struct drm_gem_object *obj = &buf->base;
	struct drm_device *dev = obj->dev;

	WARN_ON(buf->buffer_copy != NULL);

	DRM_INFO("Paging out buffer of size %zu\n", buf->base.size);

	buf->buffer_copy = vmalloc(buf->base.size);

	if (buf->buffer_copy == NULL)
		return false;

	finish_cma_pool_dma_memcpy(buf);

	memcpy(buf->buffer_copy,
	       vc4_bo_get_vaddr(&buf->base),
	       buf->base.size);

	list_del(&buf->mru_buffers_head);
	list_del(&buf->offset_node.head);

	/* Invalidate any user-space mappings */
	drm_vma_node_unmap(&obj->vma_node, dev->anon_inode->i_mapping);

	return true;
}

static bool is_unmovable_buffer(struct vc4_dev *vc4,
				struct vc4_bo *bo)
{
	/* Don’t page out the binning buffer */
	if (bo == vc4->bin_bo)
		return true;

	/* Don’t page out the dumb framebuffer */
	if (vc4->dev->fb_helper &&
	    vc4->dev->fb_helper->buffer &&
	    &bo->base == vc4->dev->fb_helper->buffer->gem)
		return true;

	return false;
}

static bool page_out_buffers_for_insertion(struct vc4_dev *vc4,
					   size_t size,
					   size_t *offset_out,
					   struct list_head **prev_out)
{
	struct vc4_bo *buffer, *tmp;
	struct drm_gem_object *fb_buf = NULL;

	if (vc4->dev->fb_helper && vc4->dev->fb_helper->buffer)
		fb_buf = vc4->dev->fb_helper->buffer->gem;

	/* We couldn’t find a big enough free slot so start paging out
	 * the least-recently used buffers until we make a slot.
	 */
	list_for_each_entry_safe_reverse(buffer,
					 tmp,
					 &vc4->cma_pool.mru_buffers,
					 mru_buffers_head) {
		struct list_head *prev;
		size_t offset;
		size_t next_offset;

		/* Don’t page out buffers that are in use */
		if (refcount_read(&buffer->usecnt))
			continue;

		/* Don’t page out special buffers used internally by
		 * the driver
		 */
		if (is_unmovable_buffer(vc4, buffer))
			continue;

		if (buffer->offset_node.head.prev ==
		    &vc4->cma_pool.offset_nodes) {
			prev = &vc4->cma_pool.offset_nodes;
			offset = 0;
		} else {
			struct vc4_offset_node *prev_node =
				container_of(buffer->offset_node.head.prev,
					     struct vc4_offset_node,
					     head);
			offset = prev_node->offset + prev_node->size;
			prev = &prev_node->head;
		}

		if (!page_out_buffer(buffer))
			continue;

		if (prev->next == &vc4->cma_pool.offset_nodes) {
			next_offset = VC4_CMA_POOL_SIZE;
		} else {
			struct vc4_offset_node *next_node =
				container_of(prev->next,
					     struct vc4_offset_node,
					     head);
			next_offset = next_node->offset;
		}

		if (next_offset - offset >= size) {
			*offset_out = offset;
			*prev_out = prev;
			return true;
		}
	}

	return false;
}

static bool get_insertion_point(struct vc4_dev *drv,
				size_t size,
				size_t *offset_out,
				struct list_head **prev_out)
{
	struct list_head *prev = &drv->cma_pool.offset_nodes;
	struct vc4_offset_node *node, *tmp;
	size_t offset = 0;

	lockdep_assert_held(&drv->bo_lock);

	list_for_each_entry_safe(node,
				 tmp,
				 &drv->cma_pool.offset_nodes,
				 head) {
		if (check_finished_cma_stub_node(node))
			continue;

		if (node->offset - offset >= size) {
			*offset_out = offset;
			*prev_out = prev;
			return true;
		}

		prev = &node->head;
		offset = node->offset + node->size;
	}

	/* There still might be enough space after the last buffer, or
	 * the pool might be empty
	 */
	if (offset + size <= VC4_CMA_POOL_SIZE) {
		*offset_out = offset;
		*prev_out = prev;
		return true;
	}

	return false;
}

static bool get_insertion_point_or_free(struct vc4_dev *drv,
					size_t size,
					size_t *offset_out,
					struct list_head **prev_out)
{
	/* Check if there is a gap already available */
	if (get_insertion_point(drv, size, offset_out, prev_out))
		return true;

	/*
	 * Not enough CMA memory in the pool, purge the userspace BO
	 * cache and retry.
	 * This is sub-optimal since we purge the whole userspace BO
	 * cache which forces user that want to re-use the BO to
	 * restore its initial content.
	 * Ideally, we should purge entries one by one and retry after
	 * each to see if CMA allocation succeeds. Or even better, try
	 * to find an entry with at least the same size.
	 */
	vc4_bo_userspace_cache_purge(drv);

	if (get_insertion_point(drv, size, offset_out, prev_out))
		return true;

	/* Try paging out some unused buffers */
	if (page_out_buffers_for_insertion(drv,
					   size,
					   offset_out,
					   prev_out))
		return true;

	DRM_INFO("Couldn't find insertion point for buffer of size %zu\n",
		 size);

	return false;
}

static bool page_in_buffer(struct vc4_dev *vc4,
			   struct vc4_bo *bo)
{
	size_t offset;
	struct list_head *prev;

	lockdep_assert_held(&vc4->bo_lock);

	DRM_INFO("Paging in buffer of size %zu\n", bo->base.size);

	if (!get_insertion_point_or_free(vc4, bo->base.size, &offset, &prev))
		return false;

	memcpy((uint8_t *) vc4->cma_pool.vaddr + offset,
	       bo->buffer_copy,
	       bo->base.size);

	bo->offset_node.offset = offset;
	vfree(bo->buffer_copy);
	bo->buffer_copy = NULL;

	list_add(&bo->offset_node.head, prev);
	list_add(&bo->mru_buffers_head, &vc4->cma_pool.mru_buffers);

	return true;
}

static int use_bo_unlocked(struct vc4_bo *bo)
{
	struct vc4_dev *vc4 = to_vc4_dev(bo->base.dev);

	lockdep_assert_held(&vc4->bo_lock);

	printk("@@@ use %p\n", bo);

	if (bo->buffer_copy) {
		if (!page_in_buffer(vc4, bo))
			return -ENOMEM;
	} else {
		/* Move the buffer to the head of the MRU list */
		list_del(&bo->mru_buffers_head);
		list_add(&bo->mru_buffers_head, &vc4->cma_pool.mru_buffers);

		finish_cma_pool_dma_memcpy(bo);
	}

	return 0;
}

/**
 * vc4_gem_create_object - Implementation of driver->gem_create_object.
 * @dev: DRM device
 * @size: Size in bytes of the memory the object will reference
 *
 * This lets the CMA helpers allocate object structs for us, and keep
 * our BO stats correct.
 */
struct drm_gem_object *vc4_create_object(struct drm_device *dev, size_t size)
{
	struct vc4_dev *vc4 = to_vc4_dev(dev);
	struct vc4_bo *bo;
	size_t offset;
	struct list_head *prev;

	mutex_lock(&vc4->bo_lock);

	if (!get_insertion_point_or_free(vc4, size, &offset, &prev)) {
		bo = ERR_PTR(-ENOMEM);
		goto out;
	}

	bo = kzalloc(sizeof(*bo), GFP_KERNEL);
	if (!bo) {
		bo = ERR_PTR(-ENOMEM);
		goto out;
	}

	bo->offset_node.offset = offset;
	bo->offset_node.size = size;
	bo->offset_node.type = VC4_OFFSET_NODE_TYPE_BUFFER;
	bo->stub_offset_node.type = VC4_OFFSET_NODE_TYPE_STUB;
	bo->buffer_copy = NULL;
	list_add(&bo->offset_node.head, prev);
	list_add(&bo->mru_buffers_head, &vc4->cma_pool.mru_buffers);

	bo->madv = VC4_MADV_WILLNEED;
	refcount_set(&bo->usecnt, 0);
	mutex_init(&bo->madv_lock);
	bo->label = VC4_BO_TYPE_KERNEL;
	vc4->bo_labels[VC4_BO_TYPE_KERNEL].num_allocated++;
	vc4->bo_labels[VC4_BO_TYPE_KERNEL].size_allocated += size;

out:
	mutex_unlock(&vc4->bo_lock);

	return IS_ERR(bo) ? (void *) bo : &bo->base;
}

struct vc4_bo *vc4_bo_create(struct drm_device *dev, size_t unaligned_size,
			     bool allow_unzeroed, enum vc4_kernel_bo_type type)
{
	size_t size = roundup(unaligned_size, PAGE_SIZE);
	struct vc4_dev *vc4 = to_vc4_dev(dev);
	struct drm_gem_object *gem_obj;
	struct vc4_bo *bo;
	int ret;

	if (size == 0)
		return ERR_PTR(-EINVAL);

	gem_obj = dev->driver->gem_create_object(dev, size);
	if (!gem_obj)
		return ERR_PTR(-ENOMEM);

	bo = to_vc4_bo(gem_obj);

	ret = drm_gem_object_init(dev, gem_obj, size);
	if (ret)
		goto error;

	ret = drm_gem_create_mmap_offset(gem_obj);
	if (ret)
		goto error;

	printk("@@@ create %p %zu%s\n",
	       bo,
	       size,
	       type == VC4_BO_TYPE_BIN ?
	       " BIN" :
	       (type == VC4_BO_TYPE_DUMB &&
		(vc4->dev->fb_helper == NULL ||
		 vc4->dev->fb_helper->buffer == NULL)) ?
	       " FBCON" :
	       "");

	/* By default, BOs do not support the MADV ioctl. This will be enabled
	 * only on BOs that are exposed to userspace (V3D, V3D_SHADER and DUMB
	 * BOs).
	 */
	bo->madv = __VC4_MADV_NOTSUPP;

	mutex_lock(&vc4->bo_lock);
	vc4_bo_set_label(&bo->base, type);
	mutex_unlock(&vc4->bo_lock);

	return bo;

error:
	vc4_bo_destroy(bo);
	return ERR_PTR(ret);
}

int vc4_dumb_create(struct drm_file *file_priv,
		    struct drm_device *dev,
		    struct drm_mode_create_dumb *args)
{
	int min_pitch = DIV_ROUND_UP(args->width * args->bpp, 8);
	struct vc4_bo *bo = NULL;
	int ret;

	if (args->pitch < min_pitch)
		args->pitch = min_pitch;

	if (args->size < args->pitch * args->height)
		args->size = args->pitch * args->height;

	bo = vc4_bo_create(dev, args->size, false, VC4_BO_TYPE_DUMB);
	if (IS_ERR(bo))
		return PTR_ERR(bo);

	bo->madv = VC4_MADV_WILLNEED;

	ret = drm_gem_handle_create(file_priv, &bo->base, &args->handle);
	drm_gem_object_put_unlocked(&bo->base);

	return ret;
}

/* Called on the last userspace/kernel unreference of the BO.
 */
void vc4_free_object(struct drm_gem_object *gem_bo)
{
	struct vc4_bo *bo = to_vc4_bo(gem_bo);

	/* Remove the BO from the purgeable list. */
	mutex_lock(&bo->madv_lock);
	if (bo->madv == VC4_MADV_DONTNEED && !refcount_read(&bo->usecnt))
		vc4_bo_remove_from_purgeable_pool(bo);
	mutex_unlock(&bo->madv_lock);

	vc4_bo_destroy(bo);
}

static bool prep_cma_pool_dma_memcpy(struct vc4_dev *vc4,
				     size_t dst_address,
				     struct vc4_bo *bo)
{
	struct dma_chan *chan = vc4->cma_pool.dma_chan;
	struct dma_async_tx_descriptor *tx;
	size_t src_address = bo->offset_node.offset;
	size_t size = bo->base.size;
	int ret;

	if (chan == NULL)
		return false;

	/* The DMA memcpy won’t work if the address range overlaps and
	 * the destination is at a higher address.
	 */
	if (dst_address >= src_address && src_address + size > dst_address)
		return false;

	/* If the source and dest ranges overlap then only mark the
	 * range after the dest as a stub to keep the data structure
	 * coherent.
	 */
	if (src_address < dst_address + size) {
		bo->stub_offset_node.offset = dst_address + size;
		bo->stub_offset_node.size = src_address - dst_address;
	} else {
		bo->stub_offset_node.offset = src_address;
		bo->stub_offset_node.size = size;
	}

	tx = chan->device->device_prep_dma_memcpy(chan,
						  vc4->cma_pool.paddr +
						  dst_address,
						  vc4->cma_pool.paddr +
						  src_address,
						  size,
						  0 /* flags */);
	if (!tx) {
		DRM_ERROR("Failed to set up DMA memcpy for the CMA pool\n");
		return false;
	}

	bo->cma_pool_dma_cookie = tx->tx_submit(tx);
	ret = dma_submit_error(bo->cma_pool_dma_cookie);
	if (ret) {
		DRM_ERROR("Failed to submit DMA: %d\n", ret);
		return false;
	}

	list_add(&bo->stub_offset_node.head, &bo->offset_node.head);

	return true;
}

static void vc4_bo_try_compact(struct vc4_bo *bo)
{
	struct vc4_dev *vc4 = to_vc4_dev(bo->base.dev);
	size_t prev_address;

	/* This is called from dec_usecnt with the madv_lock as the
	 * last decrement, so we know that the buffer isn’t paged out
	 * or purged.
	 */

	if (is_unmovable_buffer(vc4, bo))
		return;

	mutex_lock(&vc4->bo_lock);

	while (true) {
		struct vc4_offset_node *prev_node;

		if (bo->offset_node.head.prev == &vc4->cma_pool.offset_nodes) {
			prev_address = 0;
			break;
		}

		prev_node = container_of(bo->offset_node.head.prev,
					 struct vc4_offset_node,
					 head);

		if (!check_finished_cma_stub_node(prev_node)) {
			prev_address = prev_node->offset + prev_node->size;
			break;
		}
	}

	if (prev_address < bo->offset_node.offset) {
		/* Invalidate any user-space mappings */
		drm_vma_node_unmap(&bo->base.vma_node,
				   bo->base.dev->anon_inode->i_mapping);

		if (!prep_cma_pool_dma_memcpy(vc4, prev_address, bo)) {
			memmove(vc4->cma_pool.vaddr + prev_address,
				vc4->cma_pool.vaddr + bo->offset_node.offset,
				bo->base.size);
		}

		bo->offset_node.offset = prev_address;
	}

	mutex_unlock(&vc4->bo_lock);
}

int vc4_bo_inc_usecnt(struct vc4_bo *bo)
{
	int ret;

	/* Fast path: if the BO is already retained by someone, no need to
	 * check the madv status.
	 */
	if (refcount_inc_not_zero(&bo->usecnt))
		return 0;

	ret = vc4_bo_use(bo);
	if (ret)
		return ret;

	mutex_lock(&bo->madv_lock);
	switch (bo->madv) {
	case VC4_MADV_WILLNEED:
		if (!refcount_inc_not_zero(&bo->usecnt)) {
			printk("@@@ add_usecnt %p\n", bo);
			refcount_set(&bo->usecnt, 1);
		}
		ret = 0;
		break;
	case VC4_MADV_DONTNEED:
		/* We shouldn't use a BO marked as purgeable if at least
		 * someone else retained its content by incrementing usecnt.
		 * Luckily the BO hasn't been purged yet, but something wrong
		 * is happening here. Just throw an error instead of
		 * authorizing this use case.
		 */
	case __VC4_MADV_PURGED:
		/* We can't use a purged BO. */
	default:
		/* Invalid madv value. */
		ret = -EINVAL;
		break;
	}
	mutex_unlock(&bo->madv_lock);

	return ret;
}

void vc4_bo_dec_usecnt(struct vc4_bo *bo)
{
	/* Fast path: if the BO is still retained by someone, no need to test
	 * the madv value.
	 */
	if (refcount_dec_not_one(&bo->usecnt))
		return;

	mutex_lock(&bo->madv_lock);
	if (refcount_dec_and_test(&bo->usecnt)) {
		printk("@@@ remove_usecnt %p\n", bo);

		if (bo->madv == VC4_MADV_DONTNEED)
			vc4_bo_add_to_purgeable_pool(bo);
		vc4_bo_try_compact(bo);
	}
	mutex_unlock(&bo->madv_lock);
}

struct sg_table *vc4_prime_get_sg_table(struct drm_gem_object *obj)
{
	struct sg_table *sgt;
	int ret;

	sgt = kzalloc(sizeof(*sgt), GFP_KERNEL);
	if (!sgt)
		return ERR_PTR(-ENOMEM);

	ret = dma_get_sgtable(obj->dev->dev, sgt,
			      vc4_bo_get_vaddr(obj),
			      vc4_bo_get_paddr(obj),
			      obj->size);
	if (ret < 0)
		goto out;

	return sgt;

out:
	kfree(sgt);
	return ERR_PTR(ret);
}

struct dma_buf * vc4_prime_export(struct drm_gem_object *obj, int flags)
{
	struct vc4_bo *bo = to_vc4_bo(obj);
	struct dma_buf *dmabuf;
	int ret;

	if (bo->validated_shader) {
		DRM_DEBUG("Attempting to export shader BO\n");
		return ERR_PTR(-EINVAL);
	}

	/* Note: as soon as the BO is exported it becomes unpurgeable, because
	 * noone ever decrements the usecnt even if the reference held by the
	 * exported BO is released. This shouldn't be a problem since we don't
	 * expect exported BOs to be marked as purgeable.
	 */
	ret = vc4_bo_inc_usecnt(bo);
	if (ret) {
		DRM_ERROR("Failed to increment BO usecnt\n");
		return ERR_PTR(ret);
	}

	dmabuf = drm_gem_prime_export(obj, flags);
	if (IS_ERR(dmabuf))
		vc4_bo_dec_usecnt(bo);

	return dmabuf;
}

static int mmap_pgoff_dance(struct vm_area_struct *vma,
			    struct drm_gem_object *gem_obj)
{
	unsigned long vm_pgoff;
	int ret;

	/* This ->vm_pgoff dance is needed to make all parties happy:
	 * - dma_mmap_wc() uses ->vm_pgoff as an offset within the allocated
	 *   mem-region, hence the need to set it to zero (the value set by
	 *   the DRM core is a virtual offset encoding the GEM object-id)
	 * - the mmap() core logic needs ->vm_pgoff to be restored to its
	 *   initial value before returning from this function because it
	 *   encodes the  offset of this GEM in the dev->anon_inode pseudo-file
	 *   and this information will be used when we invalidate userspace
	 *   mappings  with drm_vma_node_unmap() (called from vc4_gem_purge()).
	 */
	vm_pgoff = vma->vm_pgoff;
	vma->vm_pgoff = 0;
	ret = dma_mmap_wc(gem_obj->dev->dev, vma,
			  vc4_bo_get_vaddr(gem_obj),
			  vc4_bo_get_paddr(gem_obj),
			  vma->vm_end - vma->vm_start);
	vma->vm_pgoff = vm_pgoff;

	return ret;
}

vm_fault_t vc4_fault(struct vm_fault *vmf)
{
	struct vm_area_struct *vma = vmf->vma;
	struct drm_gem_object *obj = vma->vm_private_data;
	struct vc4_bo *bo = to_vc4_bo(obj);
	struct vc4_dev *vc4 = to_vc4_dev(bo->base.dev);
	vm_fault_t ret = 0;
	int mmap_ret;

	/* Purged buffers can’t be paged in */
	mutex_lock(&bo->madv_lock);
	if (bo->madv == __VC4_MADV_PURGED)
		ret = VM_FAULT_SIGBUS;
	mutex_unlock(&bo->madv_lock);

	if (ret)
		return ret;

	DRM_INFO("Page fault for buffer %p %s\n",
		 bo,
		 vc4->bo_labels[bo->label].name);

	mutex_lock(&vc4->bo_lock);

	if (bo->buffer_copy) {
		DRM_INFO("Got page fault for paged out buffer %p %s\n",
			 bo,
			 vc4->bo_labels[bo->label].name);
	}

	if (use_bo_unlocked(bo))
		ret = VM_FAULT_OOM;

	mutex_unlock(&vc4->bo_lock);

	if (ret)
		return ret;

	mmap_ret = mmap_pgoff_dance(vma, obj);

	switch (mmap_ret) {
	case EAGAIN:
		DRM_INFO("Got EGAIN for mmap_pgoff_dance for %p %s\n",
			 bo,
			 vc4->bo_labels[bo->label].name);
		return VM_FAULT_NOPAGE;
	case 0:
		return VM_FAULT_NOPAGE;
	default:
		DRM_ERROR("Got unknown error %i for mmap_pgoff_dance "
			  "for %p %s\n",
			  mmap_ret,
			  bo,
			  vc4->bo_labels[bo->label].name);
		return VM_FAULT_OOM;
	}
}

int vc4_mmap(struct file *filp, struct vm_area_struct *vma)
{
	struct drm_gem_object *gem_obj;
	struct vc4_dev *vc4;
	struct vc4_bo *bo;
	int ret;

	ret = drm_gem_mmap(filp, vma);
	if (ret)
		return ret;

	gem_obj = vma->vm_private_data;
	vc4 = to_vc4_dev(gem_obj->dev);
	bo = to_vc4_bo(gem_obj);

	if (bo->validated_shader && (vma->vm_flags & VM_WRITE)) {
		DRM_DEBUG("mmaping of shader BOs for writing not allowed.\n");
		return -EINVAL;
	}

	if (bo->madv != VC4_MADV_WILLNEED) {
		DRM_DEBUG("mmaping of %s BO not allowed\n",
			  bo->madv == VC4_MADV_DONTNEED ?
			  "purgeable" : "purged");
		return -EINVAL;
	}

	ret = vc4_bo_use(bo);
	if (ret)
		return ret;

	/*
	 * Clear the VM_PFNMAP flag that was set by drm_gem_mmap(), and set the
	 * vm_pgoff (used as a fake buffer offset by DRM) to 0 as we want to map
	 * the whole buffer.
	 */
	vma->vm_flags &= ~VM_PFNMAP;

	ret = mmap_pgoff_dance(vma, gem_obj);

	if (ret)
		drm_gem_vm_close(vma);

	return ret;
}

int vc4_prime_mmap(struct drm_gem_object *obj, struct vm_area_struct *vma)
{
	struct vc4_bo *bo = to_vc4_bo(obj);

	if (bo->validated_shader && (vma->vm_flags & VM_WRITE)) {
		DRM_DEBUG("mmaping of shader BOs for writing not allowed.\n");
		return -EINVAL;
	}

	DRM_DEBUG("prime_mmap not implemented\n");
	return -EINVAL;
}

int vc4_bo_use(struct vc4_bo *bo)
{
	struct vc4_dev *vc4 = to_vc4_dev(bo->base.dev);
	int ret;

	mutex_lock(&vc4->bo_lock);

	ret = use_bo_unlocked(bo);

	mutex_unlock(&vc4->bo_lock);

	return ret;
}

void *vc4_prime_vmap(struct drm_gem_object *obj)
{
	struct vc4_bo *bo = to_vc4_bo(obj);
	int ret;

	if (bo->validated_shader) {
		DRM_DEBUG("mmaping of shader BOs not allowed.\n");
		return ERR_PTR(-EINVAL);
	}

	ret = vc4_bo_use(bo);
	if (ret)
		return ERR_PTR(ret);

	return vc4_bo_get_vaddr(obj);
}

void vc4_prime_vunmap(struct drm_gem_object *obj, void *vaddr)
{
	/* Nothing to do */
}

struct drm_gem_object *
vc4_prime_import_sg_table(struct drm_device *dev,
			  struct dma_buf_attachment *attach,
			  struct sg_table *sgt)
{
	DRM_DEBUG("prime_import_sg_table not implemented\n");
	return ERR_PTR(-EINVAL);
}

static int vc4_grab_bin_bo(struct vc4_dev *vc4, struct vc4_file *vc4file)
{
	int ret;

	if (!vc4->v3d)
		return -ENODEV;

	if (vc4file->bin_bo_used)
		return 0;

	ret = vc4_v3d_bin_bo_get(vc4, &vc4file->bin_bo_used);
	if (ret)
		return ret;

	return 0;
}

int vc4_create_bo_ioctl(struct drm_device *dev, void *data,
			struct drm_file *file_priv)
{
	struct drm_vc4_create_bo *args = data;
	struct vc4_file *vc4file = file_priv->driver_priv;
	struct vc4_dev *vc4 = to_vc4_dev(dev);
	struct vc4_bo *bo = NULL;
	int ret;

	ret = vc4_grab_bin_bo(vc4, vc4file);
	if (ret)
		return ret;

	/*
	 * We can't allocate from the BO cache, because the BOs don't
	 * get zeroed, and that might leak data between users.
	 */
	bo = vc4_bo_create(dev, args->size, false, VC4_BO_TYPE_V3D);
	if (IS_ERR(bo))
		return PTR_ERR(bo);

	bo->madv = VC4_MADV_WILLNEED;

	ret = drm_gem_handle_create(file_priv, &bo->base, &args->handle);
	drm_gem_object_put_unlocked(&bo->base);

	return ret;
}

int vc4_mmap_bo_ioctl(struct drm_device *dev, void *data,
		      struct drm_file *file_priv)
{
	struct drm_vc4_mmap_bo *args = data;
	struct drm_gem_object *gem_obj;

	gem_obj = drm_gem_object_lookup(file_priv, args->handle);
	if (!gem_obj) {
		DRM_DEBUG("Failed to look up GEM BO %d\n", args->handle);
		return -EINVAL;
	}

	/* The mmap offset was set up at BO allocation time. */
	args->offset = drm_vma_node_offset_addr(&gem_obj->vma_node);

	drm_gem_object_put_unlocked(gem_obj);
	return 0;
}

int
vc4_create_shader_bo_ioctl(struct drm_device *dev, void *data,
			   struct drm_file *file_priv)
{
	struct drm_vc4_create_shader_bo *args = data;
	struct vc4_file *vc4file = file_priv->driver_priv;
	struct vc4_dev *vc4 = to_vc4_dev(dev);
	struct vc4_bo *bo = NULL;
	int ret;

	if (args->size == 0)
		return -EINVAL;

	if (args->size % sizeof(u64) != 0)
		return -EINVAL;

	if (args->flags != 0) {
		DRM_INFO("Unknown flags set: 0x%08x\n", args->flags);
		return -EINVAL;
	}

	if (args->pad != 0) {
		DRM_INFO("Pad set: 0x%08x\n", args->pad);
		return -EINVAL;
	}

	ret = vc4_grab_bin_bo(vc4, vc4file);
	if (ret)
		return ret;

	bo = vc4_bo_create(dev, args->size, true, VC4_BO_TYPE_V3D_SHADER);
	if (IS_ERR(bo))
		return PTR_ERR(bo);

	bo->madv = VC4_MADV_WILLNEED;

	if (copy_from_user(vc4_bo_get_vaddr(&bo->base),
			     (void __user *)(uintptr_t)args->data,
			     args->size)) {
		ret = -EFAULT;
		goto fail;
	}
	/* Clear the rest of the memory from allocating from the BO
	 * cache.
	 */
	memset(vc4_bo_get_vaddr(&bo->base) + args->size, 0,
	       bo->base.size - args->size);

	bo->validated_shader = vc4_validate_shader(&bo->base);
	if (!bo->validated_shader) {
		ret = -EINVAL;
		goto fail;
	}

	/* We have to create the handle after validation, to avoid
	 * races for users to do doing things like mmap the shader BO.
	 */
	ret = drm_gem_handle_create(file_priv, &bo->base, &args->handle);

fail:
	drm_gem_object_put_unlocked(&bo->base);

	return ret;
}

/**
 * vc4_set_tiling_ioctl() - Sets the tiling modifier for a BO.
 * @dev: DRM device
 * @data: ioctl argument
 * @file_priv: DRM file for this fd
 *
 * The tiling state of the BO decides the default modifier of an fb if
 * no specific modifier was set by userspace, and the return value of
 * vc4_get_tiling_ioctl() (so that userspace can treat a BO it
 * received from dmabuf as the same tiling format as the producer
 * used).
 */
int vc4_set_tiling_ioctl(struct drm_device *dev, void *data,
			 struct drm_file *file_priv)
{
	struct drm_vc4_set_tiling *args = data;
	struct drm_gem_object *gem_obj;
	struct vc4_bo *bo;
	bool t_format;

	if (args->flags != 0)
		return -EINVAL;

	switch (args->modifier) {
	case DRM_FORMAT_MOD_NONE:
		t_format = false;
		break;
	case DRM_FORMAT_MOD_BROADCOM_VC4_T_TILED:
		t_format = true;
		break;
	default:
		return -EINVAL;
	}

	gem_obj = drm_gem_object_lookup(file_priv, args->handle);
	if (!gem_obj) {
		DRM_DEBUG("Failed to look up GEM BO %d\n", args->handle);
		return -ENOENT;
	}
	bo = to_vc4_bo(gem_obj);
	bo->t_format = t_format;

	drm_gem_object_put_unlocked(gem_obj);

	return 0;
}

/**
 * vc4_get_tiling_ioctl() - Gets the tiling modifier for a BO.
 * @dev: DRM device
 * @data: ioctl argument
 * @file_priv: DRM file for this fd
 *
 * Returns the tiling modifier for a BO as set by vc4_set_tiling_ioctl().
 */
int vc4_get_tiling_ioctl(struct drm_device *dev, void *data,
			 struct drm_file *file_priv)
{
	struct drm_vc4_get_tiling *args = data;
	struct drm_gem_object *gem_obj;
	struct vc4_bo *bo;

	if (args->flags != 0 || args->modifier != 0)
		return -EINVAL;

	gem_obj = drm_gem_object_lookup(file_priv, args->handle);
	if (!gem_obj) {
		DRM_DEBUG("Failed to look up GEM BO %d\n", args->handle);
		return -ENOENT;
	}
	bo = to_vc4_bo(gem_obj);

	if (bo->t_format)
		args->modifier = DRM_FORMAT_MOD_BROADCOM_VC4_T_TILED;
	else
		args->modifier = DRM_FORMAT_MOD_NONE;

	drm_gem_object_put_unlocked(gem_obj);

	return 0;
}

int vc4_bo_cma_pool_init(struct vc4_dev *vc4)
{
	dma_cap_mask_t dma_mask;

	vc4->cma_pool.vaddr = dma_alloc_wc(vc4->dev->dev,
					   VC4_CMA_POOL_SIZE,
					   &vc4->cma_pool.paddr,
					   GFP_KERNEL | __GFP_NOWARN);
	if (!vc4->cma_pool.vaddr) {
		DRM_INFO("Failed to allocate CMA pool memory\n");
		return -ENOMEM;
	}

	INIT_LIST_HEAD(&vc4->cma_pool.mru_buffers);
	INIT_LIST_HEAD(&vc4->cma_pool.offset_nodes);

	dma_cap_zero(dma_mask);
	dma_cap_set(DMA_MEMCPY, dma_mask);
	vc4->cma_pool.dma_chan = dma_request_chan_by_mask(&dma_mask);

	if (IS_ERR(vc4->cma_pool.dma_chan)) {
		DRM_ERROR("Failed to get dma_chan\n");
		vc4->cma_pool.dma_chan = NULL;
	}

	return 0;
}

void vc4_bo_cma_pool_destroy(struct vc4_dev *vc4)
{
	dma_free_wc(vc4->dev->dev,
		    VC4_CMA_POOL_SIZE,
		    vc4->cma_pool.vaddr,
		    vc4->cma_pool.paddr);

	if (vc4->cma_pool.dma_chan)
		dma_release_channel(vc4->cma_pool.dma_chan);
}

int vc4_bo_labels_init(struct drm_device *dev)
{
	struct vc4_dev *vc4 = to_vc4_dev(dev);
	int i;

	/* Create the initial set of BO labels that the kernel will
	 * use.  This lets us avoid a bunch of string reallocation in
	 * the kernel's draw and BO allocation paths.
	 */
	vc4->bo_labels = kcalloc(VC4_BO_TYPE_COUNT, sizeof(*vc4->bo_labels),
				 GFP_KERNEL);
	if (!vc4->bo_labels)
		return -ENOMEM;
	vc4->num_labels = VC4_BO_TYPE_COUNT;

	BUILD_BUG_ON(ARRAY_SIZE(bo_type_names) != VC4_BO_TYPE_COUNT);
	for (i = 0; i < VC4_BO_TYPE_COUNT; i++)
		vc4->bo_labels[i].name = bo_type_names[i];

	mutex_init(&vc4->bo_lock);

	vc4_debugfs_add_file(dev, "bo_stats", vc4_bo_stats_debugfs, NULL);

	return 0;
}

void vc4_bo_labels_destroy(struct drm_device *dev)
{
	struct vc4_dev *vc4 = to_vc4_dev(dev);
	int i;

	for (i = 0; i < vc4->num_labels; i++) {
		if (vc4->bo_labels[i].num_allocated) {
			DRM_ERROR("Destroying BO cache with %d %s "
				  "BOs still allocated\n",
				  vc4->bo_labels[i].num_allocated,
				  vc4->bo_labels[i].name);
		}

		if (is_user_label(i))
			kfree(vc4->bo_labels[i].name);
	}
	kfree(vc4->bo_labels);
}

int vc4_label_bo_ioctl(struct drm_device *dev, void *data,
		       struct drm_file *file_priv)
{
	struct vc4_dev *vc4 = to_vc4_dev(dev);
	struct drm_vc4_label_bo *args = data;
	char *name;
	struct drm_gem_object *gem_obj;
	int ret = 0, label;

	if (!args->len)
		return -EINVAL;

	name = strndup_user(u64_to_user_ptr(args->name), args->len + 1);
	if (IS_ERR(name))
		return PTR_ERR(name);

	gem_obj = drm_gem_object_lookup(file_priv, args->handle);
	if (!gem_obj) {
		DRM_ERROR("Failed to look up GEM BO %d\n", args->handle);
		kfree(name);
		return -ENOENT;
	}

	mutex_lock(&vc4->bo_lock);
	label = vc4_get_user_label(vc4, name);
	if (label != -1)
		vc4_bo_set_label(gem_obj, label);
	else
		ret = -ENOMEM;
	mutex_unlock(&vc4->bo_lock);

	drm_gem_object_put_unlocked(gem_obj);

	return ret;
}

dma_addr_t vc4_bo_get_paddr(struct drm_gem_object *obj)
{
	struct vc4_dev *vc4 = to_vc4_dev(obj->dev);

	return vc4->cma_pool.paddr + to_vc4_bo(obj)->offset_node.offset;
}

void *vc4_bo_get_vaddr(struct drm_gem_object *obj)
{
	struct vc4_dev *vc4 = to_vc4_dev(obj->dev);

	return vc4->cma_pool.vaddr + to_vc4_bo(obj)->offset_node.offset;
}
