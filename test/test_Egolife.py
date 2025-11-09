from datasets import load_dataset

def test_load_egolife_dataset():
    ds = load_dataset("lmms-lab/EgoLife")
    print(type(ds))

if __name__ == "__main__":
    test_load_egolife_dataset()