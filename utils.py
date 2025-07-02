def frame_to_seconds(frame_number, fps=None):
    """
    Chuyển đổi từ frame number sang giây
    
    Args:
        frame_number: Số frame
        fps: Frames per second (nếu None sẽ dùng DEFAULT_FPS)
    
    Returns:
        Thời gian tính bằng giây
    """
    return frame_number / fps


def seconds_to_frame(seconds, fps=None):
    """
    Chuyển đổi từ giây sang frame number
    
    Args:
        seconds: Thời gian tính bằng giây
        fps: Frames per second (nếu None sẽ dùng DEFAULT_FPS)
    
    Returns:
        Số frame
    """
    return int(seconds * fps)


def extract_keyframe_info(keyframe_name):
    """
    Trích xuất thông tin lesson, video và frame từ tên keyframe
    
    Args:
        keyframe_name: Tên file keyframe (vd: L01_V001_000260.jpg)
        
    Returns:
        Tuple chứa lesson, video, frame (vd: ('L01', 'V001', '000260'))
    """
    parts = keyframe_name.split('_')
    return parts[0], parts[1], parts[2].split('.')[0]