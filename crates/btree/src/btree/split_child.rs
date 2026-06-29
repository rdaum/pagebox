use pagebox_storage::buffer_pool::ExclusiveFrame;
use pagebox_swip_kernel::SwipWord as Swip;

use super::node::ResidentFrame;

/// ```anneal
/// ensures: ret.val = frame_pid.val
/// proof (h_anon):
///   unfold split_child_frame_pid at h_returns
///   cases h_returns
///   rfl
/// proof (h_progress):
///   unfold split_child_frame_pid
///   simp_all
/// ```
pub(crate) fn split_child_frame_pid(frame_pid: u64) -> u64 {
    frame_pid
}

#[derive(Clone, Copy)]
pub(crate) struct SplitChild<'frame, 'guard> {
    frame: &'frame ExclusiveFrame<'guard>,
}

impl<'frame, 'guard> SplitChild<'frame, 'guard> {
    pub(crate) fn from_exclusive(frame: &'frame ExclusiveFrame<'guard>) -> Self {
        Self { frame }
    }

    pub(crate) fn resident_frame(self) -> ResidentFrame {
        ResidentFrame::from_exclusive(self.frame)
    }

    pub(crate) fn swip(self) -> Swip {
        self.resident_frame().hot_swip()
    }

    pub(crate) fn pid(self) -> u64 {
        split_child_frame_pid(self.frame.pid())
    }
}
