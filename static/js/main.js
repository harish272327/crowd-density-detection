document.addEventListener('DOMContentLoaded', function () {
  const video = document.getElementById('webcam');
  const captureBtn = document.getElementById('captureBtn');
  const startCameraBtn = document.getElementById('startCameraBtn');
  const snapshotCanvas = document.getElementById('snapshotCanvas');
  const webcamDataInput = document.getElementById('webcam_data');

  if (!video || !captureBtn || !startCameraBtn || !snapshotCanvas || !webcamDataInput) {
    return;
  }

  let stream = null;

  startCameraBtn.addEventListener('click', async function () {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      video.srcObject = stream;
      video.play();
    } catch (error) {
      alert('Unable to access webcam: ' + error.message);
    }
  });

  captureBtn.addEventListener('click', function () {
    if (!stream) {
      alert('Please start the webcam first.');
      return;
    }

    const context = snapshotCanvas.getContext('2d');
    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;
    snapshotCanvas.width = width;
    snapshotCanvas.height = height;
    context.drawImage(video, 0, 0, width, height);

    const dataUrl = snapshotCanvas.toDataURL('image/jpeg', 0.9);
    webcamDataInput.value = dataUrl;
    alert('Webcam frame captured successfully. Submit the form to analyze it.');
  });
});
