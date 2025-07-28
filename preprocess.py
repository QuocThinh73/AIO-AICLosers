import argparse
import os
import sys


def shot_boundary_detection(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["all", "lesson"])
    parser.add_argument("input_video_dir", type=str)
    parser.add_argument("output_shot_dir", type=str)
    parser.add_argument("--lesson_name", type=str)
    
    args = parser.parse_args(argv)
    
    # Check error
    if not os.path.exists(args.input_video_dir):
        raise ValueError("Input video directory does not exist")
    
    if args.mode == "lesson":
        if not args.lesson_name:
            raise ValueError("Lesson name is required when mode is lesson")
        if not os.path.exists(os.path.join(args.input_video_dir, args.lesson_name)):
            raise ValueError("Lesson video directory does not exist")
        
    # Main process
    from preprocess.shot_boundary_detection import detect_shot_boundary
    if args.mode == "all":
        detect_shot_boundary(args.input_video_dir, args.output_shot_dir, args.mode)
    elif args.mode == "lesson":
        detect_shot_boundary(args.input_video_dir, args.output_shot_dir, args.mode, args.lesson_name)
    
def keyframe_extraction(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["all", "lesson"])
    parser.add_argument("input_video_dir", type=str)
    parser.add_argument("input_shot_dir", type=str)
    parser.add_argument("output_keyframe_dir", type=str)
    parser.add_argument("--lesson_name", type=str)
    
    args = parser.parse_args(argv)
    
    # Check error
    if not os.path.exists(args.input_video_dir):
        raise ValueError("Input video directory does not exist")
    
    if not os.path.exists(args.input_shot_dir):
        raise ValueError("Input shot directory does not exist")
    
    if args.mode == "lesson":
        if not args.lesson_name:
            raise ValueError("Lesson name is required when mode is lesson")
        if not os.path.exists(os.path.join(args.input_video_dir, args.lesson_name)):
            raise ValueError("Lesson video's directory does not exist")
        if not os.path.exists(os.path.join(args.input_shot_dir, args.lesson_name)):
            raise ValueError("Lesson shot's directory does not exist")
        
    # Main process
    from preprocess.keyframe_extraction import extract_keyframe
    if args.mode == "all":
        extract_keyframe(args.input_video_dir, args.input_shot_dir, args.output_keyframe_dir, args.mode)
    elif args.mode == "lesson":
        extract_keyframe(args.input_video_dir, args.input_shot_dir, args.output_keyframe_dir, args.mode, args.lesson_name)

def build_mapping_json(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("input_keyframe_dir", type=str)
    parser.add_argument("output_mapping_json", type=str)

    args = parser.parse_args(argv)
    
    # Check error
    if not os.path.exists(args.input_keyframe_dir):
        raise ValueError("Input keyframe directory does not exist")
    
    # Main process
    from preprocess.build_mapping_json import build_mapping_json
    build_mapping_json(args.input_keyframe_dir, args.output_mapping_json)

def news_anchor_detection(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["all", "lesson"])
    parser.add_argument("input_keyframe_dir", type=str)
    parser.add_argument("output_news_anchor_dir", type=str)
    parser.add_argument("--lesson_name", type=str)

    args = parser.parse_args(argv)
    
    # Check error
    if not os.path.exists(args.input_keyframe_dir):
        raise ValueError("Input keyframe directory does not exist")
    
    if args.mode == "lesson":
        if not args.lesson_name:
            raise ValueError("Lesson name is required when mode is lesson")
        if not os.path.exists(os.path.join(args.input_keyframe_dir, args.lesson_name)):
            raise ValueError("Lesson keyframe's directory does not exist")
    
    # Main process
    from preprocess.news_anchor_detection import detect_news_anchor
    if args.mode == "all":
        detect_news_anchor(args.input_keyframe_dir, args.output_news_anchor_dir, args.mode)
    elif args.mode == "lesson":
        detect_news_anchor(args.input_keyframe_dir, args.output_news_anchor_dir, args.mode, args.lesson_name)

def news_segmentation(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["all", "lesson"])
    parser.add_argument("input_keyframe_dir", type=str)
    parser.add_argument("input_news_anchor_dir", type=str)
    parser.add_argument("output_news_segment_dir", type=str)
    parser.add_argument("--lesson_name", type=str)

    args = parser.parse_args(argv)
    
    # Check error
    if not os.path.exists(args.input_keyframe_dir):
        raise ValueError("Input keyframe directory does not exist")
    
    if not os.path.exists(args.input_news_anchor_dir):
        raise ValueError("Input news anchor directory does not exist")
    
    if args.mode == "lesson":
        if not args.lesson_name:
            raise ValueError("Lesson name is required when mode is lesson")
        if not os.path.exists(os.path.join(args.input_keyframe_dir, args.lesson_name)):
            raise ValueError("Lesson keyframe's directory does not exist")
        if not os.path.exists(os.path.join(args.input_news_anchor_dir, args.lesson_name)):
            raise ValueError("Lesson news anchor's directory does not exist")
    
    # Main process
    from preprocess.news_segmentation import segment_news
    if args.mode == "all":
        segment_news(args.input_keyframe_dir, args.input_news_anchor_dir, args.output_news_segment_dir, args.mode)
    elif args.mode == "lesson":
        segment_news(args.input_keyframe_dir, args.input_news_anchor_dir, args.output_news_segment_dir, args.mode, args.lesson_name)

def subvideo_extraction(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["all", "lesson"])
    parser.add_argument("input_video_dir", type=str)
    parser.add_argument("input_segment_dir", type=str)
    parser.add_argument("output_subvideo_dir", type=str)
    parser.add_argument("ffmpeg_bin", type=str)
    parser.add_argument("--lesson_name", type=str)
    
    args = parser.parse_args(argv)
    
    # Check error
    if not os.path.exists(args.input_video_dir):
        raise ValueError("Input video directory does not exist")
    
    if not os.path.exists(args.input_segment_dir):
        raise ValueError("Input segment directory does not exist")
    
    if not args.ffmpeg_bin:
        raise ValueError("FFmpeg binary path is required")
    
    if not os.path.exists(args.ffmpeg_bin):
        raise ValueError("FFmpeg binary does not exist at the specified path")
    
    if args.mode == "lesson":
        if not args.lesson_name:
            raise ValueError("Lesson name is required when mode is lesson")
        if not os.path.exists(os.path.join(args.input_video_dir, args.lesson_name)):
            raise ValueError("Lesson video's directory does not exist")
        if not os.path.exists(os.path.join(args.input_segment_dir, args.lesson_name)):
            raise ValueError("Lesson segment's directory does not exist")
    
    # Main process
    from preprocess.subvideo_extraction import extract_subvideo
    if args.mode == "all":
        extract_subvideo(args.input_video_dir, args.input_segment_dir, args.output_subvideo_dir, args.mode, args.ffmpeg_bin)
    elif args.mode == "lesson":
        extract_subvideo(args.input_video_dir, args.input_segment_dir, args.output_subvideo_dir, args.mode, args.ffmpeg_bin, args.lesson_name)

def remove_noise_keyframe(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["all", "lesson"])
    parser.add_argument("keyframe_dir", type=str)
    parser.add_argument("--lesson_name", type=str)
    
    args = parser.parse_args(argv)
    
    # Check error TODO
        
    # Main process TODO
    from preprocess.remove_noise_keyframe import remove_noise_keyframe

def object_detection(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["all", "lesson", "single"])
    parser.add_argument("input_keyframe_dir", type=str)
    parser.add_argument("input_caption_dir", type=str)
    parser.add_argument("output_detection_dir", type=str)
    parser.add_argument("--lesson_name", type=str)
    parser.add_argument("--video_name", type=str)
    
    args = parser.parse_args(argv)
    
    # Check error
    if not os.path.exists(args.input_keyframe_dir):
        raise ValueError("Input keyframe directory does not exist")
    
    if not os.path.exists(args.input_caption_dir):
        raise ValueError("Input caption directory does not exist")
    
    if args.mode == "lesson":
        if not args.lesson_name:
            raise ValueError("Lesson name is required when mode is lesson")
        if not os.path.exists(os.path.join(args.input_keyframe_dir, args.lesson_name)):
            raise ValueError(f"Lesson keyframe directory does not exist: {os.path.join(args.input_keyframe_dir, args.lesson_name)}")
        if not os.path.exists(os.path.join(args.input_caption_dir, args.lesson_name)):
            raise ValueError(f"Lesson caption directory does not exist: {os.path.join(args.input_caption_dir, args.lesson_name)}")
    
    elif args.mode == "single":
        if not args.lesson_name:
            raise ValueError("Lesson name is required when mode is single")
        if not args.video_name:
            raise ValueError("Video name is required when mode is single")
        
        lesson_keyframe_dir = os.path.join(args.input_keyframe_dir, args.lesson_name)
        if not os.path.exists(lesson_keyframe_dir):
            raise ValueError(f"Lesson keyframe directory does not exist: {lesson_keyframe_dir}")
        
        video_keyframe_dir = os.path.join(lesson_keyframe_dir, args.video_name)
        if not os.path.exists(video_keyframe_dir):
            raise ValueError(f"Video keyframe directory does not exist: {video_keyframe_dir}")
        
        caption_file = os.path.join(args.input_caption_dir, args.lesson_name, 
                                  f"{args.lesson_name}_{args.video_name}_caption.json")
        if not os.path.exists(caption_file):
            raise ValueError(f"Caption file does not exist: {caption_file}")
    
    # Main process
    from preprocess.object_detection import detect_object
    
    if args.mode == "all":
        detect_object(args.input_keyframe_dir, args.input_caption_dir, args.output_detection_dir, args.mode)
    elif args.mode == "lesson":
        detect_object(args.input_keyframe_dir, args.input_caption_dir, args.output_detection_dir, args.mode, args.lesson_name)
    elif args.mode == "single":
        detect_object(args.input_keyframe_dir, args.input_caption_dir, args.output_detection_dir, args.mode, args.lesson_name, args.video_name)

def image_captioning(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["all", "lesson", "single"])
    parser.add_argument("--input_keyframe_dir", type=str)
    parser.add_argument("--output_caption_dir", type=str)
    parser.add_argument("--lesson_name", type=str)
    parser.add_argument("--video_name", type=str)
    
    args = parser.parse_args(argv)
    
    # Check error
    if not os.path.exists(args.input_keyframe_dir):
        raise ValueError("Input keyframe directory does not exist")
    
    os.makedirs(args.output_caption_dir, exist_ok=True)
    
    if args.mode == "lesson":
        if not args.lesson_name:
            raise ValueError("Lesson name is required when mode is lesson")
        lesson_keyframe_dir = os.path.join(args.input_keyframe_dir, args.lesson_name)
        if not os.path.exists(lesson_keyframe_dir):
            raise ValueError(f"Lesson keyframe directory does not exist: {lesson_keyframe_dir}")
    
    elif args.mode == "single":
        if not args.lesson_name:
            raise ValueError("Lesson name is required when mode is single")
        if not args.video_name:
            raise ValueError("Video name is required when mode is single")
        
        lesson_keyframe_dir = os.path.join(args.input_keyframe_dir, args.lesson_name)
        if not os.path.exists(lesson_keyframe_dir):
            raise ValueError(f"Lesson keyframe directory does not exist: {lesson_keyframe_dir}")
        
        video_keyframe_dir = os.path.join(lesson_keyframe_dir, args.video_name)
        if not os.path.exists(video_keyframe_dir):
            raise ValueError(f"Video keyframe directory does not exist: {video_keyframe_dir}")
    
    # Main process
    from preprocess.image_captioning import generate_captions
    
    if args.mode == "all":
        generate_captions(args.input_keyframe_dir, args.output_caption_dir, args.mode)
    elif args.mode == "lesson":
        generate_captions(args.input_keyframe_dir, args.output_caption_dir, args.mode, args.lesson_name)
    elif args.mode == "single":
        generate_captions(args.input_keyframe_dir, args.output_caption_dir, args.mode, args.lesson_name, args.video_name)
    
    # Image captioning process completed

def asr(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["all", "lesson"])
    parser.add_argument("--lesson_name", type=str)
    
    args = parser.parse_args(argv)
    
    # Check error TODO
    
    # Main process TODO
    from preprocess.asr import transcribe_audio
    
def ocr(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["all", "lesson"])
    parser.add_argument("--lesson_name", type=str)
    
    args = parser.parse_args(argv)
    
    # Check error TODO
    
    # Main process TODO
    from preprocess.ocr import extract_text
    

TASKS = {
    "shot_boundary_detection": shot_boundary_detection,
    "keyframe_extraction": keyframe_extraction,
    "build_mapping_json": build_mapping_json,
    "news_anchor_detection": news_anchor_detection,
    "news_segmentation": news_segmentation,
    "subvideo_extraction": subvideo_extraction,
    "remove_noise_keyframe": remove_noise_keyframe,
    "object_detection": object_detection,
    "image_captioning": image_captioning,
    "asr": asr,
    "ocr": ocr,
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <task> [options]")
        print("Available tasks:")
        for task in TASKS:
            print(f"- {task}")
        sys.exit(1)
    
    task = sys.argv[1]
    argv = sys.argv[2:]
    
    if task in TASKS:
        TASKS[task](argv)
    else:
        print(f"Task '{task}' not found. Available tasks:")
        for task in TASKS:
            print(f"- {task}")
        sys.exit(1)      