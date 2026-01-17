from esm.models.esm3 import ESM3

def download_weights():
    print("Downloading ESM-3 sm_open_v1 weights...")
    # This will trigger the download if weights are not present
    model = ESM3.from_pretrained("esm3_sm_open_v1")
    print("Download complete!")

if __name__ == "__main__":
    download_weights()
