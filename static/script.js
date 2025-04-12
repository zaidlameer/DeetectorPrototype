document.getElementById("upload-form").addEventListener("submit", async function(event) {
    event.preventDefault();

    const fileInput = document.getElementById("video-input");
    const loadingDiv = document.getElementById("loading");
    const resultsDiv = document.getElementById("results");
    const audioResult = document.getElementById("audio-result");
    const videoResult = document.getElementById("video-result");
    const audioConfidence = document.getElementById("audio-confidence");
    const videoConfidence = document.getElementById("video-confidence");
    const uploadedVideo = document.getElementById("uploaded-video");

    if (fileInput.files.length === 0) {
        alert("Please select a video file.");
        return;
    }

    const formData = new FormData();
    const file = fileInput.files[0];
    formData.append("video", file);

    loadingDiv.classList.remove("hidden");
    resultsDiv.classList.add("hidden");

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (result.error) {
            alert("Error: " + result.error);
        } else {
            // Set result text
            audioResult.textContent = result.audio_result;
            videoResult.textContent = result.video_result;

            // Set confidence text
            audioConfidence.textContent = result.audio_confidence + "%";
            videoConfidence.textContent = result.video_confidence + "%";

            // Show the uploaded video
            uploadedVideo.src = URL.createObjectURL(file);
            uploadedVideo.style.display = "block";

            resultsDiv.classList.remove("hidden");
        }
    } catch (error) {
        alert("Failed to process the video.");
    } finally {
        loadingDiv.classList.add("hidden");
    }
});

document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.getElementById('theme-toggle');
    const body = document.body;

    // Save theme preference
    if (localStorage.getItem('theme') === 'dark') {
        body.classList.add('dark-mode');
        toggle.textContent = '☀️';
    }

    toggle.addEventListener('click', () => {
        body.classList.toggle('dark-mode');
        if (body.classList.contains('dark-mode')) {
            toggle.textContent = '☀️';
            localStorage.setItem('theme', 'dark');
        } else {
            toggle.textContent = '🌙';
            localStorage.setItem('theme', 'light');
        }
    });
});


document.addEventListener('DOMContentLoaded', () => {
    document.body.classList.add('loaded');
});
