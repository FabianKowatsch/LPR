// Function to select an example image
function selectExample(element) {
    const exampleSelect = document.getElementById("exampleSelect");
    const exampleImages = document.querySelectorAll(".example-img");

    // Set the hidden input value to the selected example's filename
    exampleSelect.value = element.getAttribute("data-filename");

    // Highlight the selected image
    exampleImages.forEach((img) => img.classList.remove("selected"));
    element.classList.add("selected");
}



let licensePlates = [];
class LicensePlate {
    constructor(lpText, filteredText) {
        this.lpText = lpText;  // License plate text
        this.filteredText = filteredText;
        this.images = [];

        this.boundingBoxes = [];  // Array to store bounding boxes
        this.frames = [];      // Array to store frames where the license plate is detected
    }

    // Add a frame and corresponding bounding box
    addDetection(image, box, frame) {
        this.images.push(image);
        this.boundingBoxes.push(box);
        this.frames.push(frame);
    }

    // Get the latest frame and bounding box
    getLatestDetection() {
        if (this.frames.length > 0) {
            return {
                image: this.images[this.images.length - 1],
                frame: this.frames[this.frames.length - 1],
                bbox: this.boundingBoxes[this.boundingBoxes.length - 1],
            };
        }
        return null;
    }
}

// Function to add a license plate to the list, if it already exists add the frame and bbox
function addLicensePlate(plateItem) {
    let existingLicensePlate = null;

    const threshold = 2; // Allow up to 2 character differences
    // Check if the license plate already exists in the array
    for (let lp of licensePlates) {
        // Calculate the Levenshtein distance between the current plate and the existing one
        const distance = levenshtein(lp.lpText, plateItem.lp_text);
        // If the distance is below the threshold, consider them as the same plate
        if (distance <= threshold) {
            existingLicensePlate = lp;
            break;
        }
    }

    // If license plate exists, add the frame and bbox
    if (existingLicensePlate) {
        existingLicensePlate.addDetection(plateItem.image, plateItem.box, plateItem.frame || 0);
    } else {
        // Otherwise, create a new LicensePlate object and add it to the list
        const newLicensePlate = new LicensePlate(plateItem.lp_text, plateItem.text_filtered);
        newLicensePlate.addDetection(plateItem.image, plateItem.box, plateItem.frame || 0);
        licensePlates.push(newLicensePlate);
    }
}


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
    const resultsList = document.getElementById("resultsList");

    // Hide the upload container and show the result container
    uploadContainer.style.display = "none";
    resultContainer.style.display = "flex";

    // Clear previous content
    imageContainer.innerHTML = "";
    resultsList.innerHTML = "";

    // Determine if the uploaded file is a video
    const videoExtensions = [".mp4", ".avi", ".mov", ".mkv"];
    const fileExtension = results.filename.split('.').pop().toLowerCase();
    const isVideo = videoExtensions.includes(`.${fileExtension}`);

    let mediaElement = null;
    let bboxOverlay = null;

    if (isVideo) {
        // Create a <video> element
        mediaElement = document.createElement("video");
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

    const frameRate = results.results[0]?.fps || 30;
    // Loop through inference results and create result items
    licensePlates.forEach((plate) => {
        console.log("Number of images:", plate.images.length);
        const resultItem = document.createElement("div");
        resultItem.classList.add("result-item");

        // Create an image element for the cropped license plate
        const croppedImg = document.createElement("img");
        croppedImg.src = plate.images[0];
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
            if (isVideo && mediaElement) {
                // Calculate the time from the frame number
                const frameNumber = plate.frames[0]; // The frame number you want to seek to

                // Calculate the exact time in seconds based on the frame and fps
                const timeInSeconds = frameNumber / frameRate;

                // Set the video to the exact time
                mediaElement.currentTime = timeInSeconds;
            }
        });

        // mediaElement.addEventListener("loadedmetadata", () => {
        highlightBoundingBoxes(licensePlates, bboxOverlay, mediaElement, frameRate)
        // });

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

        console.log(media, mediaRect, media.naturalWidth, media.naturalHeight, media.videoWidth, media.videoHeight);

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
            console.log(previousIndex, nextIndex);

            if (nextIndex === null || previousIndex === null) {
                return;
            }

            const previousFrame = plate.frames[previousIndex];
            const nextFrame = plate.frames[nextIndex];

            const distance = Math.abs(nextFrame - previousFrame);
            const currentDistance = Math.abs(currentFrame - previousFrame);
            const time = currentDistance / distance;

            const previousBbox = parseBbox(plate.boundingBoxes[previousIndex][0]);
            const nextBbox = parseBbox(plate.boundingBoxes[nextIndex][0]);

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
                console.log(plates[i].lpText);
                const bbox = parseBbox(plates[i].boundingBoxes[0][0]);
                showBoundingBox(bbox, plates[i].filteredText);
            }
        })
    }
}


// UTILS_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
// Function to calculate Levenshtein distance
function levenshtein(a, b) {
    const tmp = [];
    for (let i = 0; i <= b.length; i++) tmp[i] = [i];
    for (let j = 0; j <= a.length; j++) tmp[0][j] = j;

    for (let i = 1; i <= b.length; i++) {
        for (let j = 1; j <= a.length; j++) {
            tmp[i][j] = Math.min(
                tmp[i - 1][j] + 1,        // Deletion
                tmp[i][j - 1] + 1,        // Insertion
                tmp[i - 1][j - 1] + (a[j - 1] === b[i - 1] ? 0 : 1)  // Substitution
            );
        }
    }

    return tmp[b.length][a.length];
}

function parseBbox(box) {
    return {
        x: box[0],
        y: box[1],
        width: box[2] - box[0],
        height: box[3] - box[1]
    };
}

function interpolateBoundingBoxes(bbox1, bbox2, t) {
    return {
        x: bbox1.x + (bbox2.x - bbox1.x) * t,
        y: bbox1.y + (bbox2.y - bbox1.y) * t,
        width: bbox1.width + (bbox2.width - bbox1.width) * t,
        height: bbox1.height + (bbox2.height - bbox1.height) * t
    };
}

