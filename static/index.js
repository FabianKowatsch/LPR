let frameRate = 30;
let isVideo = false;

function runInference() {
    const fileInput = document.getElementById("fileInput");
    const exampleSelect = document.getElementById("exampleSelect");
    const recognizerSelect = document.getElementById("recognizerSelect");
    // const frameInterval = document.getElementById("frameInterval");
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
    const frameIntervalValue = 1; //frameInterval.value || 1;

    let localImageUrl = null;
    if (file) localImageUrl = file ? URL.createObjectURL(file) : null;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("example", examplePath);
    formData.append("recognizer", recognizer);
    formData.append("frameInterval", frameIntervalValue);

    // Progress nur bei Video anzeigen
    if (file && file.type.startsWith("video/")) {
        document.getElementById("progressContainer").style.display = "block";
    } else {
        document.getElementById("progressContainer").style.display = "none";
    }

    // Send POST request to /upload endpoint
    fetch("/upload", {
        method: "POST",
        body: formData,
    })
        .then((response) => response.json())
        .then((data) => {
            // check if the response contains an error message
            if (data.error) {
                alert(data.error);
                return;
            }
            console.log("Upload success:", data);
            processResults({
                file_url: localImageUrl || data.file_url,
                filename: data.filename,
                results: data.results,
            });
        })
        .catch((error) => {
            console.error("Upload error:", error);
        });

}

function processResults(results) {
    document.getElementById("progressContainer").style.display = "none";
    const uploadContainer = document.getElementById("uploadContainer");
    const resultContainer = document.getElementById("resultContainer");
    const imageContainer = document.getElementById("imageContainer");
    const resultsList = document.getElementById("resultsList");

    // Hide the upload container and show the result container
    uploadContainer.style.display = "none";
    resultContainer.style.display = "flex";

    // Clear previous content
    imageContainer.innerHTML = "";
    resultsList.innerHTML = "";

    // Determine if the uploaded file is a video
    const videoExtensions = [".mp4", ".avi", ".mov", ".mkv"];
    const fileExtension = results.filename.split(".").pop().toLowerCase();
    isVideo = videoExtensions.includes(`.${fileExtension}`);

    let mediaElement = null;
    let videoPlayer = null;
    let bboxOverlay = null;

    if (isVideo) {
        // Create a <video> element
        mediaElement = document.createElement("video-js");
        mediaElement.classList.add("video-js");
        mediaElement.id = "videoElement";
        mediaElement.style = `
            max-width: 100%; 
            max-height: 100%; 
            display: block; 
            margin: auto;
        `;

        const source = document.createElement("source");
        source.src = results.file_url;
        source.type = "video/mp4";
        mediaElement.appendChild(source);

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
            display: flex;
            width: 100%;
            height: auto;
        `;
        mediaWrapper.appendChild(mediaElement);
        mediaWrapper.appendChild(bboxOverlay);

        imageContainer.appendChild(mediaWrapper);

        videoPlayer = videojs('videoElement', {
            controls: true,
            autoplay: false,
            preload: 'auto',
            muted: true,
            aspectRatio: '16:9',
            controlBar: {
                volumePanel: false,  // Hides the audio controls (mute/unmute)
                fullscreenToggle: false,  // Hides the fullscreen button
            },
        });
    } else {
        // Create a container for image and bounding boxes
        const mediaWrapper = document.createElement("div");
        mediaWrapper.style = `
            position: relative; 
            display: inline-block;
        `;

        // Create the original image
        mediaElement = document.createElement("img");
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

        // Delete search bar
    }

    // // Back to Home Button hinzufügen
    // const backButton = document.createElement("a");
    // backButton.href = "/";
    // backButton.className = "btn btn-primary w-100 mt-4";
    // backButton.textContent = " ← Upload Another File";
    // resultsList.appendChild(backButton);

    // Mention Filename and Recognizer
    const recognizerSelect = document.getElementById("recognizerSelect");
    const recognizer = recognizerSelect.value;
    const headline = document.createElement("h4");
    headline.className = "result-headline";
    headline.innerHTML = `Detection complete: "<span class="filename">${results.filename}</span>" processed by <span class="recognizer">${recognizer}</span>.`;
    resultsList.appendChild(headline);

    // Add license plate detection results
    results.results.forEach((item) => {
        if (item["box_raw"] !== undefined) {
            rawBoxes.push(item)
        } else {
            addLicensePlate(item);
        }
    })
    findHigestConfidenceText();
    filterLicensePlates();
    joinLicensePlates();

    if (results.results[0]?.fps) {
        frameRate = results.results[0]?.fps
    }

    // Loop through inference results and create result items
    showLicensePlateList(licensePlates);
    highlightBoundingBoxes(licensePlates, bboxOverlay, mediaElement, videoPlayer);
}

function showLicensePlateList(plates) {
    const resultsList = document.getElementById("resultsList");
    if (isVideo) {
        resultsList.innerHTML = `<input type="text" id="searchBar" class="search-bar" placeholder="Search for License Plate"
        onchange="searchResults(this)">`;   
    }

    plates.forEach((plate) => {
        // Wir merken uns, ob wir mindestens eine gültige Plate angezeigt haben
        let hasValidPlate = false;
        // 1. Prüfen, ob plate.filteredText existiert und nicht leer ist
        if (!plate.filteredText || plate.filteredText.trim() === "") {
            return; // Überspringen
        }
        // 2. Prüfen, ob das Kennzeichen nur eine Fehlermeldung ist (falls du so etwas abfängst)
        //    z. B. "OCR processing failed: No valid text detected"
        //    Wenn deine Fehlertexte immer gleich aufgebaut sind, kannst du z. B. so filtern:
        if (plate.filteredText.includes("OCR processing failed")) {
            return; // Überspringen
        }
        // Wenn wir hier sind, haben wir ein valides Kennzeichen.
        hasValidPlate = true;

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
            const videoPlayer = videojs('videoElement');
            if (videoPlayer) {
                // Calculate the time from the frame number
                const frameNumber = plate.frames[0]; // Use the first frame

                // Calculate the exact time in seconds based on the frame and fps
                const timeInSeconds = frameNumber / frameRate;

                // Set the video to the exact time
                videoPlayer.currentTime(timeInSeconds);
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
function highlightBoundingBoxes(plates, overlay, media, videoPlayer) {
    const toggleBoxes = document.getElementById("toggleBoxes");

    // let firstFrameTime = null;
    let mediaRect = null;
    let scaleX = 1;
    let scaleY = 1;
    let frameOffset = 0;
    let frameCallbackId = null;

    function getMediaScale() {
        if (videoPlayer) {
            mediaRect = videoPlayer.el().getBoundingClientRect();
            scaleX = mediaRect.width / (videoPlayer.videoWidth() || videoPlayer.el().videoWidth);
            scaleY = mediaRect.height / (videoPlayer.videoHeight() || videoPlayer.el().videoHeight);
        } else {
            mediaRect = media.getBoundingClientRect();
            scaleX = mediaRect.width / (media.naturalWidth);
            scaleY = mediaRect.height / (media.naturalHeight);
        }
    }

    function showBoundingBox(bbox, plateID, boxColor) {
        // Create bounding box container
        const bboxContainer = document.createElement("div");
        bboxContainer.style = `
            position: absolute;
            left: ${bbox.x * scaleX}px;
            top: ${bbox.y * scaleY}px;
            width: ${bbox.width * scaleX}px;
            height: ${bbox.height * scaleY}px;
            outline: 2px solid ${boxColor}; 
            background: rgba(191, 10, 70, 0.1);
            pointer-events: none;
        `;

        // Create label
        const label = document.createElement("div");
        label.innerText = plateID;
        label.style = `
            position: absolute;
            left: -1.5px; /* Align with outline */
            top: -28px; /* Adjust based on outline */
            width: fit-content;
            padding: 2px;
            background: ${boxColor};
            color: white;
            font-weight: bold;
            text-align: center;
            white-space: nowrap; /* Prevent text wrapping */
            overflow: hidden;
            text-overflow: ellipsis;
            text-align: center;
        `;

        // Append label to bounding box
        bboxContainer.appendChild(label);
        overlay.appendChild(bboxContainer);
    }

    function showFirstBoundingBoxes() {
        for (let i = 0; i < plates.length; i++) {
            const bbox = parseBbox(plates[i].boundingBoxes[0][0]);
            showBoundingBox(bbox, plates[i].filteredText, "rgb(191, 10, 70)");
        }
    }

    function showCurrentBoundingBoxes() {
        overlay.innerHTML = ""; // Clear previous bounding boxes

        let currentTime = videoPlayer.currentTime();
        const currentFrame = Math.max(0, Math.round(currentTime * frameRate) - frameOffset);
        // console.log(`Current time: ${media.currentTime}, frameRate: ${frameRate}, Current frame: ${currentFrame}`);

        if (toggleBoxes.checked) {
            plates.forEach((plate, index) => {
                if (currentFrame < plate.frames[0]
                    || currentFrame > plate.frames[plate.frames.length - 1]
                ) {
                    return;
                }

                let bbox = null;
                let color = "rgb(191, 10, 70)";
                const currentFrameIndex = plate.frames.indexOf(currentFrame);
                if (currentFrameIndex === -1) {
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

                    if (nextFrame - previousFrame > 4) { // if more than 4 frames between dont interpolate!
                        return;
                    }

                    const distance = Math.abs(nextFrame - previousFrame);
                    const currentDistance = Math.abs(currentFrame - previousFrame);
                    const time = currentDistance / distance;

                    const previousBbox = parseBbox(plate.boundingBoxes[previousIndex]);
                    const nextBbox = parseBbox(plate.boundingBoxes[nextIndex]);

                    color = plate.isTracked[previousIndex] ? color : "orange";
                    bbox = interpolateBoundingBoxes(previousBbox, nextBbox, time);
                } else {
                    color = plate.isTracked[currentFrameIndex] ? color : "orange";
                    bbox = parseBbox(plate.boundingBoxes[currentFrameIndex]);
                }
                
                // const color = plate.isTracked ? "rgb(191, 10, 70)" : "blue";
                showBoundingBox(bbox, plate.filteredText, color);
            });
        } else {
            rawBoxes.forEach((rawBox, index) => {
                const frame = Math.floor(rawBox["frame"]);
                if (frame == currentFrame) {
                    const bbox = parseBbox(rawBox["box_raw"]);
                    showBoundingBox(bbox, index, "green");
                }
            });
        }
    }

    let animationFrameId = null;
    let previousFrame = 0;
    function updateBoundingBoxes() {
        if (!videoPlayer || videoPlayer.paused() || videoPlayer.ended()) {
            return; // Stop updates if video is not playing
        }

        const currentTime = videoPlayer.currentTime();
        const currentFrame = Math.floor(currentTime * frameRate);
        // console.log(`Current time: ${currentTime}, frameRate: ${frameRate}, Current frame: ${currentFrame}`);
        if (currentFrame !== previousFrame) { // Only update if the frame changed
            previousFrame = currentFrame;
            showCurrentBoundingBoxes();
        }

        animationFrameId = requestAnimationFrame(updateBoundingBoxes);
    }

    if (videoPlayer) {
        videoPlayer.on('play', () => {
            getMediaScale();
            animationFrameId = requestAnimationFrame(updateBoundingBoxes);
        });

        videoPlayer.on('pause', () => {
            showCurrentBoundingBoxes();
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
                animationFrameId = null;
            }
        });

        videoPlayer.on('seeked', () => {
            if (videoPlayer.paused()) {
                showCurrentBoundingBoxes();
            }
        });
        videoPlayer.on('timeupdate', () => {
            if (videoPlayer.paused()) {
                showCurrentBoundingBoxes();
            }
        });

        toggleBoxes.addEventListener("change", () => {
            overlay.innerHTML = "";
            showCurrentBoundingBoxes();
        });

    } else {
        media.addEventListener('load', () => {
            getMediaScale();
            showFirstBoundingBoxes();
        })

        // Turn off the toggle container if media is image
        const toggleContainer = document.getElementById("toggleContainer");
        toggleContainer.style.display = "none";
    }

    window.addEventListener('resize', () => {
        getMediaScale();
        if (videoPlayer) {
            showCurrentBoundingBoxes();
        } else {
            overlay.innerHTML = ""; // Clear previous bounding boxes
            showFirstBoundingBoxes();
        }
    });
}

function searchResults(searchBar) {
    const plates = searchLicensePlates(searchBar.value)
    showLicensePlateList(plates);
}