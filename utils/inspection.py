import pickle
with open("/cluster/project/cvg/students/tnanni/ego4d_data/v2/full_validation/6deb79d2-b2af-4afe-b097-b98a66c0f703_encoded.pkl", "rb") as f:
  file = pickle.load(f)

dp = file.video_datapoints[0]
print(dp.video_path)
print(dp.scenes["scene_0"])
