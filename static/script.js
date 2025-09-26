const video = document.getElementById('video');
const captureBtn = document.getElementById('capture');
const canvas = document.getElementById('canvas');
const imageDataInput = document.getElementById('image_data');
const cameraForm = document.getElementById('cameraForm');

// Access the camera
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error("Error accessing camera:", err);
    });

// Capture & compress image
captureBtn.addEventListener('click', () => {
    const context = canvas.getContext('2d');

    // Set smaller resolution (e.g., 640x480)
    const targetWidth = 640;
    const targetHeight = 480;

    canvas.width = targetWidth;
    canvas.height = targetHeight;

    // Draw scaled image
    context.drawImage(video, 0, 0, targetWidth, targetHeight);

    // Convert to JPEG with 50% quality
    const dataURL = canvas.toDataURL('image/jpeg', 0.5);

    // Set hidden input value
    imageDataInput.value = dataURL;

    // Submit form
    cameraForm.submit();
});