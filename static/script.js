document.getElementById("upload-form").addEventListener("submit", async function(event) {
    event.preventDefault();
    
    const fileInput = document.getElementById("video-input");
    const loadingDiv = document.getElementById("loading");
    const resultsDiv = document.getElementById("results");
    const audioResult = document.getElementById("audio-result");
    const videoResult = document.getElementById("video-result");
    
    if (fileInput.files.length === 0) {
        alert("Please select a video file.");
        return;
    }
    
    const formData = new FormData();
    formData.append("video", fileInput.files[0]);
    
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
            resultsDiv.classList.remove("hidden");
        }
    } catch (error) {
        alert("Failed to process the video.");
    } finally {
        loadingDiv.classList.add("hidden");
    }
});
