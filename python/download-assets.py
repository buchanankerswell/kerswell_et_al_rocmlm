from rocml import download_and_unzip

# OSF assets folder url
url = (
    "https://files.osf.io/v1/resources/k23tb/providers/osfstorage/"
    "649149796513ba03733a3536/?zip="
)

# OSF doi
doi = "doi.org/10.17605/OSF.IO/K23TB"

# Download all assets from OSF
print(f"Downloading assets from: {doi}")

download_and_unzip(url, "assets")

print("download-assets.py done!")