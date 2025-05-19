document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("video-input");
    const clearButton = document.getElementById("clear-button");
    const loadingDiv = document.getElementById("loading");
    const resultsDiv = document.getElementById("results");
    const audioResult = document.getElementById("audio-result");
    const videoResult = document.getElementById("video-result");
    const audioConfidence = document.getElementById("audio-confidence");
    const videoConfidence = document.getElementById("video-confidence");
    const uploadedVideo = document.getElementById("uploaded-video");

    document.getElementById("upload-form").addEventListener("submit", async function (event) {
        event.preventDefault();

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
                audioResult.textContent = result.audio_result;
                videoResult.textContent = result.video_result;
                audioConfidence.textContent = result.audio_confidence + "%";
                videoConfidence.textContent = result.video_confidence + "%";

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

    clearButton.addEventListener("click", () => {
        fileInput.value = "";

        uploadedVideo.src = "";
        uploadedVideo.style.display = "none";

        resultsDiv.classList.add("hidden");

        audioResult.textContent = "";
        videoResult.textContent = "";
        audioConfidence.textContent = "";
        videoConfidence.textContent = "";
    });

    const toggle = document.getElementById("theme-toggle");
    const body = document.body;

    if (localStorage.getItem("theme") === "dark") {
        body.classList.add("dark-mode");
        toggle.textContent = "☀️";
    }

    toggle.addEventListener("click", () => {
        body.classList.toggle("dark-mode");
        if (body.classList.contains("dark-mode")) {
            toggle.textContent = "☀️";
            localStorage.setItem("theme", "dark");
        } else {
            toggle.textContent = "🌙";
            localStorage.setItem("theme", "light");
        }
    });

    document.body.classList.add("loaded");
});
