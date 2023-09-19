from rocml import download_and_unzip

# Download all assets from OSF
url = "https://files.osf.io/v1/resources/k23tb/providers/osfstorage/649149796513ba03733a3536/?zip="

print("Downloading assets from:")
print(f"    {url}")

download_and_unzip(url, "assets")