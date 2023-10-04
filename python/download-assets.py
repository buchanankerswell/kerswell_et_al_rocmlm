from scripting import download_and_unzip

# OSF assets folder url
assets_url = "https://files.osf.io/v1/resources/k23tb/providers/osfstorage/649149796513ba03733a3536/?zip="

# Download all assets from OSF
print("Downloading assets from: doi.org/10.17605/OSF.IO/K23TB")

download_and_unzip(assets_url, "assets")

print("download-assets.py done!")