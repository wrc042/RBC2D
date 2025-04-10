param (
    [string]$dir,
    [string]$name
)

if (-not (Test-Path $dir)) {
    Write-Host "The directory '$dir' does not exist."
    exit 1
}

# Construct the ffmpeg command
$ffmpegCommand = "ffmpeg -framerate 24 -i `"$dir/img/$name/$name-%04d.png`" -vf `"scale=trunc(iw/2)*2:trunc(ih/2)*2`" -c:v libx264 -pix_fmt yuv420p `"$name.mp4`""

# Execute the ffmpeg command
Invoke-Expression $ffmpegCommand