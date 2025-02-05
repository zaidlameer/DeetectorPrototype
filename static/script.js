document.getElementById("upload-form").addEventListener("submit", function(event) {
    event.preventDefault();

    let fileInput = document.getElementById("video-file");
    let file = fileInput.files[0];

    if (!file) {
        alert("Please select a video file.");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    let loadingDiv = document.getElementById("loading");
    let resultDiv = document.getElementById("result");
    loadingDiv.classList.remove("hidden");
    resultDiv.innerHTML = "";

    fetch("/predict_video", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        loadingDiv.classList.add("hidden");
        if (data.error) {
            resultDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
        } else {
            resultDiv.innerHTML = `
                <p><strong>Final Prediction:</strong> ${data.final_prediction}</p>
                <p><strong>Audio Prediction:</strong> ${data.audio_prediction} (${(data.audio_probability * 100).toFixed(2)}%)</p>
                <p><strong>Video Prediction:</strong> ${data.video_prediction} (${(data.video_probability * 100).toFixed(2)}%)</p>
            `;
        }
    })
    .catch(error => {
        loadingDiv.classList.add("hidden");
        resultDiv.innerHTML = `<p style="color: red;">Error: ${error}</p>`;
    });
});
