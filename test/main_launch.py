from launch import Launcher

def main():
    launcher = Launcher(
        dataset_type="ego4d",
        video_path="../../ego4d_data/v2/full_scale/",
        nlq_annotations_path="../../ego4d_data/v2/annotations/nlq_train.json",
        generator=None,  # Replace with actual LLM generator instance
        modalities=["video", "audio", "text", "caption"],
        topk_videos=3,
        topk_scenes=5,
        is_pickle=False,
        save_dir=None,
        save_encoded=False,
        workers=2
    )

    launcher.run()

if __name__ == "__main__":
    main()