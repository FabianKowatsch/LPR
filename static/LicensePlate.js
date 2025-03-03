let licensePlates = [];
class LicensePlate {
    constructor(trackID) {
        this.trackID = trackID;
        this.lpText = "";  // License plate text
        this.filteredText = "";
        this.image = null;
        this.maxConfidence = 0;

        this.lpTexts = [];
        this.filteredTexts = [];
        this.images = [];
        this.confidences = [];

        this.boundingBoxes = [];
        this.frames = [];      // Array to store frames where the license plate is detected
    }

    // Add a frame and corresponding bounding box
    addDetection(lpText, filteredText, confidence, image, box, frame) {
        this.lpTexts.push(lpText);
        this.filteredTexts.push(filteredText);
        this.confidences.push(confidence);
        this.images.push(image);

        this.boundingBoxes.push(box);
        this.frames.push(frame);
    }

    joinLicensePlate(licensePlate) {
        if (this.maxConfidence < licensePlate.maxConfidence) {
            this.lpText = licensePlate.lpText;
            this.filteredText = licensePlate.filteredText;
            this.image = licensePlate.image;
            this.maxConfidence = licensePlate.maxConfidence;
        }

        this.boundingBoxes = this.boundingBoxes.concat(licensePlate.boundingBoxes);
        this.frames = this.frames.concat(licensePlate.frames);
    }

    findHigestConfidenceText() {
        let highestConfidence = 0;
        let highestConfidenceIndex = 0;

        // get the median length of the license plate texts strings
        const medianLength = this.lpTexts.reduce((a, b) => a + b.length, 0) / this.lpTexts.length;

        // search for the highest confidence text, filter out the ones that are significantly shorter
        for (let i = 0; i < this.confidences.length; i++) {
            if (this.confidences[i] > highestConfidence &&
                this.lpTexts[i].length > medianLength * 0.75 &&
                isValidPlate(this.lpTexts[i])) {
                highestConfidence = this.confidences[i];
                highestConfidenceIndex = i;
            }
        }

        this.lpText = this.lpTexts[highestConfidenceIndex];
        this.filteredText = this.filteredTexts[highestConfidenceIndex];
        this.image = this.images[highestConfidenceIndex];
        this.maxConfidence = highestConfidence;
    }
}

// Function to add a license plate to the list, if it already exists add the frame and bbox
function addLicensePlate(plateItem) {
    const trackID = plateItem.track_id || licensePlates.length + 1;

    // Check if the license plate already exists in the array
    let existingLicensePlate = null;
    for (let lp of licensePlates) {
        if (trackID === lp.trackID) {
            existingLicensePlate = lp;
            break;
        }
    }

    // If license plate exists, add the frame and bbox
    if (existingLicensePlate) {
        existingLicensePlate.addDetection(
            plateItem.lp_text,
            plateItem.text_filtered,
            plateItem.confidence || 1,
            plateItem.image,
            plateItem.box,
            plateItem.frame || 0
        );
    } else {
        // Otherwise, create a new LicensePlate object and add it to the list
        if (!plateItem.error) {
            const newLicensePlate = new LicensePlate(trackID);
            newLicensePlate.addDetection(
                plateItem.lp_text,
                plateItem.text_filtered,
                plateItem.confidence || 1,
                plateItem.image,
                plateItem.box,
                plateItem.frame || 0
            );
            licensePlates.push(newLicensePlate);
        } else {
            const newLicensePlate = new LicensePlate(plateItem.error);
            licensePlates.push(newLicensePlate);
        }

    }

}

function findHigestConfidenceText() {
    licensePlates.forEach((plate) => plate.findHigestConfidenceText());
}

// Filter out invalid license plates
function filterLicensePlates() {
    licensePlates = licensePlates.filter((plate) => isValidPlate(plate.lpText));
}

function joinLicensePlates() {
    // Join license plates that have a close levenshtein distance
    const threshold = 2; // Allow up to 1 character differences

    for (let i = 0; i < licensePlates.length; i++) {
        for (let j = i + 1; j < licensePlates.length; j++) {
            const distance = levenshtein(licensePlates[i].lpText, licensePlates[j].lpText);
            if (distance <= threshold) {
                licensePlates[i].joinLicensePlate(licensePlates[j]);
                licensePlates.splice(j, 1);
                j--; // Decrement j to account for the removed element
            }
        }
    }
}

function searchLicensePlates(searchText) {
    plates = licensePlates;
    // iterate over license plates and sort them by levenshtein distance to the search text
    plates.sort((a, b) => levenshtein(a.lpText, searchText) - levenshtein(b.lpText, searchText));
    return plates;
}