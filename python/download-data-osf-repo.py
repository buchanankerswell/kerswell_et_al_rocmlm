from magemin import download_and_unzip

# Download earthchem data
url = "https://files.osf.io/v1/resources/k23tb/providers/osfstorage/648855d9bee36d01130e5f39/?zip="
download_and_unzip(url, "data")