#![allow(
    unused_unsafe,
    reason = "NoLatches construction is always explicit at test call sites"
)]

use pagebox_storage::buffer_frame::BufferFrameRef;
use pagebox_storage::buffer_pool::{ExclusiveFrame, NewUnlinkedPage, PinnedFrame};
use pagebox_swip_kernel::SwipWord as Swip;

/// The two child frames of a split after their exclusive latches are released.
///
/// The retained pin keeps the HOT identity valid while parent publication may
/// block. Parent-link hints are installed by reacquiring each child latch only
/// after the parent latch has been released.
pub(crate) struct SplitChild<'pool> {
    frame: SplitChildFrame<'pool>,
}

enum SplitChildFrame<'pool> {
    Published(PinnedFrame<'pool>),
    Unpublished(NewUnlinkedPage<'pool>),
    Transitioning,
}

impl<'pool> SplitChild<'pool> {
    pub(crate) fn from_exclusive(frame: ExclusiveFrame<'pool>) -> Self {
        Self {
            frame: SplitChildFrame::Published(frame.into_pinned()),
        }
    }

    pub(crate) fn from_unlinked(frame: NewUnlinkedPage<'pool>) -> Self {
        Self {
            frame: SplitChildFrame::Unpublished(frame),
        }
    }

    fn pinned(&self) -> &PinnedFrame<'pool> {
        match &self.frame {
            SplitChildFrame::Published(frame) => frame,
            SplitChildFrame::Unpublished(_) => {
                panic!("unpublished split child has no independently cloneable pin")
            }
            SplitChildFrame::Transitioning => unreachable!(),
        }
    }

    pub(crate) fn frame_ref(&self) -> BufferFrameRef {
        match &self.frame {
            SplitChildFrame::Published(frame) => unsafe { frame.frame_ref() },
            SplitChildFrame::Unpublished(frame) => unsafe { frame.frame_ref() },
            SplitChildFrame::Transitioning => unreachable!(),
        }
    }

    pub(crate) fn clone_pin(&self) -> PinnedFrame<'pool> {
        self.pinned().clone_pin()
    }

    pub(crate) fn swip(&self) -> Swip {
        match &self.frame {
            SplitChildFrame::Published(frame) => frame.hot_swip(),
            SplitChildFrame::Unpublished(frame) => frame.hot_swip(),
            SplitChildFrame::Transitioning => unreachable!(),
        }
    }

    pub(crate) fn pid(&self) -> u64 {
        match &self.frame {
            SplitChildFrame::Published(frame) => frame.pid(),
            SplitChildFrame::Unpublished(frame) => frame.pid(),
            SplitChildFrame::Transitioning => unreachable!(),
        }
    }

    /// Complete the unpublished split-child transition after the caller has
    /// installed a parent/root/sibling edge that makes it reachable.
    ///
    /// # Safety
    /// The structural edge must already be published.
    pub(crate) unsafe fn mark_published(&mut self) {
        let SplitChildFrame::Unpublished(_) = &self.frame else {
            return;
        };
        let unpublished = match std::mem::replace(&mut self.frame, SplitChildFrame::Transitioning) {
            SplitChildFrame::Unpublished(frame) => frame,
            _ => unreachable!(),
        };
        self.frame = SplitChildFrame::Published(unsafe { unpublished.finish_publication() });
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use pagebox_storage::buffer_pool::{BufferPool, NoLatches};

    use super::SplitChild;

    #[test]
    fn split_child_releases_latch_but_retains_pin() {
        let pool = BufferPool::new(2);
        let edge = pool.allocate_page();
        let frame = pool.fix_stable(&edge, unsafe { NoLatches::new(&pool) });
        let child = SplitChild::from_exclusive(frame.exclusive());

        assert_eq!(child.clone_pin().pid(), child.pid());
        assert!(
            child.clone_pin().try_exclusive().is_ok(),
            "parent publication must not retain the split child's exclusive latch"
        );
    }

    #[test]
    fn split_and_delete_acquisition_order_cannot_form_the_captured_cycle() {
        let pool = BufferPool::new(3);
        let left_edge = pool.allocate_page();
        let right_edge = pool.allocate_page();
        let parent_edge = pool.allocate_page();
        let initial_latches_held = Arc::new(Barrier::new(2));
        let split_children_released = Arc::new(Barrier::new(2));

        std::thread::scope(|scope| {
            let split_pool = &pool;
            let split_left_edge = &left_edge;
            let split_right_edge = &right_edge;
            let split_parent_edge = &parent_edge;
            let initial = Arc::clone(&initial_latches_held);
            let released = Arc::clone(&split_children_released);
            let split = scope.spawn(move || {
                let left =
                    split_pool.fix_stable(split_left_edge, unsafe { NoLatches::new(split_pool) });
                let right =
                    split_pool.fix_stable(split_right_edge, unsafe { NoLatches::new(split_pool) });
                let left = left.exclusive();
                let right = right.exclusive();
                initial.wait();

                let left = SplitChild::from_exclusive(left);
                let right = SplitChild::from_exclusive(right);
                released.wait();

                // This may block on delete's parent latch, but no child latch
                // is retained while it does so.
                let parent = split_pool
                    .fix_stable(split_parent_edge, unsafe { NoLatches::new(split_pool) })
                    .exclusive();
                drop(parent);
                (left.pid(), right.pid())
            });

            let delete_pool = &pool;
            let delete_right_edge = &right_edge;
            let delete_parent_edge = &parent_edge;
            let initial = Arc::clone(&initial_latches_held);
            let released = Arc::clone(&split_children_released);
            let delete = scope.spawn(move || {
                let parent = delete_pool
                    .fix_stable(delete_parent_edge, unsafe { NoLatches::new(delete_pool) })
                    .exclusive();
                initial.wait();
                released.wait();

                let sibling = delete_pool
                    .try_fix_stable(delete_right_edge)
                    .expect("split sibling must remain resident while pinned")
                    .try_exclusive()
                    .unwrap_or_else(|_| {
                        panic!("split must release the sibling latch before waiting on parent")
                    });
                drop(sibling);
                drop(parent);
            });

            delete.join().expect("delete model thread panicked");
            assert_eq!(
                split.join().expect("split model thread panicked"),
                (left_edge.page_id(), right_edge.page_id())
            );
        });
    }
}
