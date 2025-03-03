let frameRate = 30;
// Functions to upload file and process the results_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
function runInference() {
    const fileInput = document.getElementById("fileInput");
    const exampleSelect = document.getElementById("exampleSelect");
    const recognizerSelect = document.getElementById("recognizerSelect");
    const frameInterval = document.getElementById("frameInterval");
    const errorMessage = document.getElementById("errorMessage");

    if (
        (!fileInput.files.length && !exampleSelect.value) ||
        !recognizerSelect.value
    ) {
        errorMessage.classList.remove("d-none");
        return;
    } else {
        errorMessage.classList.add("d-none");
    }

    const file = fileInput.files[0];
    const examplePath = exampleSelect.value || null;
    const recognizer = recognizerSelect.value;
    const frameIntervalValue = frameInterval.value || 1;

    let localImageUrl = null;
    if (file)
        localImageUrl = file ? URL.createObjectURL(file) : null;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("example", examplePath);
    formData.append("recognizer", recognizer);
    formData.append("frameInterval", frameIntervalValue);

    fetch("/upload", {
        method: "POST",
        body: formData,
    })
        .then(response => response.json())
        .then(data => {
            // check if the response contains an error message
            if (data.error) {
                alert(data.error);
                return;
            }
            console.log("Upload success:", data);
            processResults({ file_url: localImageUrl || data.file_url, filename: data.filename, results: data.results });
        })
        .catch(error => {
            console.error("Upload error:", error);
        });
}

function processResults(results) {
    const uploadContainer = document.getElementById("uploadContainer");
    const resultContainer = document.getElementById("resultContainer");
    const imageContainer = document.getElementById("imageContainer");

    // Hide the upload container and show the result container
    uploadContainer.style.display = "none";
    resultContainer.style.display = "flex";

    // Clear previous content
    imageContainer.innerHTML = "";

    // Determine if the uploaded file is a video
    const videoExtensions = [".mp4", ".avi", ".mov", ".mkv"];
    const fileExtension = results.filename.split('.').pop().toLowerCase();
    const isVideo = videoExtensions.includes(`.${fileExtension}`);

    let mediaElement = null;
    let bboxOverlay = null;

    if (isVideo) {
        // Create a <video> element
        mediaElement = document.createElement("video");
        mediaElement.id = "videoElement";
        mediaElement.src = results.file_url;
        mediaElement.controls = true;
        mediaElement.autoplay = false;
        mediaElement.style = `
            max-width: 100%; 
            height: auto; 
            max-height: 100%; 
            display: block; 
            margin: auto;
        `;

        // Create bounding box overlay for video
        bboxOverlay = document.createElement("div");
        bboxOverlay.style = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        `;

        // Wrap video and overlay
        const mediaWrapper = document.createElement("div");
        mediaWrapper.style = `
            position: relative; 
            display: inline-block;
        `;
        mediaWrapper.appendChild(mediaElement);
        mediaWrapper.appendChild(bboxOverlay);

        imageContainer.appendChild(mediaWrapper);
    } else {
        // Create a container for image and bounding boxes
        const mediaWrapper = document.createElement("div");
        mediaWrapper.style = `
            position: relative; 
            display: inline-block;
        `;

        // Create the original image
        mediaElement = document.createElement("img");
        mediaElement.id = "imageElement";
        mediaElement.src = results.file_url;
        mediaElement.alt = results.filename;
        mediaElement.style = `
            max-width: 100%; 
            height: auto; 
            max-height: 100%;
        `;
        mediaWrapper.appendChild(mediaElement);

        // Create bounding box overlay
        bboxOverlay = document.createElement("div");
        bboxOverlay.style = `
            position: absolute; 
            top: 0; 
            left: 0; 
            width: 100%; 
            height: 100%; 
            pointer-events: none;
        `;
        mediaWrapper.appendChild(bboxOverlay);

        imageContainer.appendChild(mediaWrapper);
    }

    // Add license plate detection results
    results.results.forEach((item) => {
        addLicensePlate(item);
    })
    findHigestConfidenceText();
    filterLicensePlates();
    joinLicensePlates();

    if (results.results[0]?.fps) {
        frameRate = results.results[0]?.fps
    }
    // Loop through inference results and create result items
    showLicensePlateList(licensePlates);
    highlightBoundingBoxes(licensePlates, bboxOverlay, mediaElement, frameRate)
}

function showLicensePlateList(plates) {
    const resultsList = document.getElementById("resultsList");
    resultsList.innerHTML = "";

    plates.forEach((plate) => {
        console.log(`Track ID: ${plate.trackID}, Plate text: ${plate.lpText}, Max Confidence: ${plate.maxConfidence}`);

        const resultItem = document.createElement("div");
        resultItem.classList.add("result-item");

        // Create an image element for the cropped license plate
        const croppedImg = document.createElement("img");
        croppedImg.src = plate.image;
        croppedImg.alt = plate.lp_text;
        croppedImg.style = `
            max-width: 100%; 
            height: 50px; 
            border-radius: 5px;
        `;
        resultItem.appendChild(croppedImg);

        // Create a text element for the detected license plate text
        const textElement = document.createElement("p");
        textElement.innerHTML = `<strong>Detected:</strong> ${plate.lpText}`;
        resultItem.appendChild(textElement);

        // Create a text element for the filtered license plate text
        const filteredTextElement = document.createElement("p");
        filteredTextElement.innerHTML = `<strong>Filtered:</strong> ${plate.filteredText}`;
        resultItem.appendChild(filteredTextElement);

        // Add event listener for clicking result items
        resultItem.style.cursor = "pointer";
        resultItem.addEventListener("click", () => {
            const videoElement = document.getElementById("videoElement");
            if (videoElement) {
                // Calculate the time from the frame number
                const frameNumber = plate.frames[0]; // The frame number you want to seek to

                // Calculate the exact time in seconds based on the frame and fps
                const timeInSeconds = frameNumber / frameRate;

                // Set the video to the exact time
                videoElement.currentTime = timeInSeconds;
            }
        });

        resultsList.appendChild(resultItem);
    });
}

/**
 * Highlights the bounding box on either an image or a video frame.
 * @param {Object} bbox - The bounding box {x, y, width, height}.
 * @param {HTMLElement} overlay - The overlay container.
 * @param {HTMLMediaElement} media - The media element (image or video).
 */
function highlightBoundingBoxes(plates, overlay, media, framerate) {
    function showBoundingBox(bbox, plateID) {
        // Get media dimensions
        const mediaRect = media.getBoundingClientRect();
        const scaleX = mediaRect.width / (media.naturalWidth || media.videoWidth);
        const scaleY = mediaRect.height / (media.naturalHeight || media.videoHeight);

        // Create bounding box container
        const bboxContainer = document.createElement("div");
        bboxContainer.style = `
            position: absolute;
            left: ${bbox.x * scaleX}px;
            top: ${bbox.y * scaleY}px;
            width: ${bbox.width * scaleX}px;
            height: ${bbox.height * scaleY}px;
            outline: 3px solid rgb(191, 10, 70); 
            background: rgba(191, 10, 70, 0.1);
            pointer-events: none;
        `;

        // Create label
        const label = document.createElement("div");
        label.innerText = plateID;
        label.style = `
            position: absolute;
            left: -3px; /* Align with outline */
            top: -28px; /* Adjust based on outline */
            width: calc(100% + 6px);
            padding: 4px;
            background: rgb(191, 10, 70);
            color: white;
            font-weight: bold;
            font-size: 14px;
            text-align: center;
        `;

        // Append label to bounding box
        bboxContainer.appendChild(label);
        overlay.appendChild(bboxContainer);
    }


    function showCurrentBoundingBoxes() {
        overlay.innerHTML = ""; // Clear previous bounding boxes
        const currentFrame = Math.ceil(media.currentTime * framerate);

        plates.forEach((plate, index) => {
            if (currentFrame < plate.frames[0]
                || currentFrame > plate.frames[plate.frames.length - 1]
            ) {
                return;
            }

            // Find the closest bounding boxes for interpolation
            let previousIndex = null;
            let nextIndex = null;

            for (let i = 0; i < plate.frames.length - 1; i++) {
                if (plate.frames[i] <= currentFrame && plate.frames[i + 1] >= currentFrame) {
                    previousIndex = i;
                    nextIndex = i + 1;
                    break;
                }
            }

            if (nextIndex === null || previousIndex === null) {
                return;
            }

            const previousFrame = plate.frames[previousIndex];
            const nextFrame = plate.frames[nextIndex];

            const distance = Math.abs(nextFrame - previousFrame);
            const currentDistance = Math.abs(currentFrame - previousFrame);
            const time = currentDistance / distance;

            const previousBbox = parseBbox(plate.boundingBoxes[previousIndex]);
            const nextBbox = parseBbox(plate.boundingBoxes[nextIndex]);

            const bbox = interpolateBoundingBoxes(previousBbox, nextBbox, time);
            showBoundingBox(bbox, plate.filteredText);
        });
    }

    let animationFrameId = null;

    function updateBoundingBoxes() {
        showCurrentBoundingBoxes();
        animationFrameId = requestAnimationFrame(updateBoundingBoxes);
    }

    if (media instanceof HTMLVideoElement) {
        media.addEventListener('play', () => {
            updateBoundingBoxes();
        });

        media.addEventListener('pause', () => {
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
                animationFrameId = null;
            }
        });

        media.addEventListener('seeked', () => {
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
                animationFrameId = null;
            }
            showCurrentBoundingBoxes();
        });

        media.addEventListener('timeupdate', () => {
            showCurrentBoundingBoxes();
        });
    } else {
        media.addEventListener('load', () => {
            for (let i = 0; i < plates.length; i++) {
                const bbox = parseBbox(plates[i].boundingBoxes[0][0]);
                showBoundingBox(bbox, plates[i].filteredText);
            }
        })
    }
}

function searchResults(searchBar) {
    const plates = searchLicensePlates(searchBar.value)
    showLicensePlateList(plates);
}

