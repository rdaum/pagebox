use pagebox_storage::buffer_frame::BufferFrameRef;
use pagebox_storage::buffer_pool::ExclusiveFrame;
use pagebox_swip_kernel::SwipWord as Swip;

use super::node::ResidentFrame;

/// Old split-child reference that borrows the ExclusiveFrame (holding a latch).
/// Used by the current split code which holds child latches during parent
/// publication. Will be replaced by SplitChildIdentity when the split lock
/// refactor is complete.
#[derive(Clone, Copy)]
pub(crate) struct SplitChild<'frame, 'guard> {
    frame: &'frame ExclusiveFrame<'guard>,
}

impl<'frame, 'guard> SplitChild<'frame, 'guard> {
    pub(crate) fn from_exclusive(frame: &'frame ExclusiveFrame<'guard>) -> Self {
        Self { frame }
    }

    pub(crate) fn resident_frame(&self) -> ResidentFrame<'guard> {
        ResidentFrame::from_exclusive(self.frame)
    }

    pub(crate) fn swip(self) -> Swip {
        self.resident_frame().hot_swip()
    }

    pub(crate) fn pid(self) -> u64 {
        self.frame.pid()
    }
}

/// Identity information for a child node that was split.
///
/// Unlike `SplitChild` which borrows `&ExclusiveFrame` (holding a latch),
/// `SplitChildIdentity` holds only Copy values: the frame's `BufferFrameRef`
/// (identity), PID, and hot SWIP. This allows the split to release the
/// exclusive latches on both children before publishing the separator to
/// the parent — the B-link sibling pointers ensure the tree remains
/// traversable during the window between split and parent publication.
#[derive(Clone, Copy)]
pub(crate) struct SplitChildIdentity {
    bf: BufferFrameRef,
    pid: u64,
    swip: Swip,
}

impl SplitChildIdentity {
    pub(crate) fn from_exclusive(frame: &ExclusiveFrame<'_>) -> Self {
        let rf = ResidentFrame::from_exclusive(frame);
        Self {
            bf: rf.bf(),
            pid: rf.pid(),
            swip: rf.hot_swip(),
        }
    }

    pub(crate) fn bf(&self) -> BufferFrameRef {
        self.bf
    }

    pub(crate) fn pid(&self) -> u64 {
        self.pid
    }

    pub(crate) fn swip(&self) -> Swip {
        self.swip
    }
}
