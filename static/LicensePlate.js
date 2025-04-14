let licensePlates = [];
let rawBoxes = [];
class LicensePlate {
    constructor(trackID) {
        this.trackID = trackID;
        // this.isTracked = isTracked;

        this.lpText = "";  // License plate text
        this.filteredText = "";
        this.image = null;
        this.maxConfidence = 0;

        this.lpTexts = [];
        this.filteredTexts = [];
        this.images = [];
        this.confidences = [];
        this.isTracked = []; // if a bounding box has been tracked using DeepSort or not

        this.boundingBoxes = [];
        this.frames = [];      // Array to store frames where the license plate is detected
    }

    // Add a frame and corresponding bounding box
    addDetection(lpText, filteredText, confidence, image, box, frame, isTracked) {
        this.lpTexts.push(lpText);
        this.filteredTexts.push(filteredText);
        this.confidences.push(confidence);
        this.images.push(image);
        this.isTracked.push(isTracked)

        this.boundingBoxes.push(box);
        this.frames.push(frame);
    }

    joinLicensePlate(licensePlate) {
        // Update lpText, filteredText, image, and maxConfidence if necessary
        if (this.maxConfidence < licensePlate.maxConfidence) {
            this.lpText = licensePlate.lpText;
            this.filteredText = licensePlate.filteredText;
            this.image = licensePlate.image;
            this.maxConfidence = licensePlate.maxConfidence;
        }
    
        // Merge frames ensuring uniqueness and sorting
        const mergedFrames = Array.from(new Set([...this.frames, ...licensePlate.frames])).sort((a, b) => a - b);
       
        // Create a mapping from frame to bounding box, lpText, filteredText, image, and confidence index
        const frameToData = {};
    
        // Add data for the current license plate
        this.frames.forEach((frame, index) => {
            frameToData[frame] = {
                boundingBox: this.boundingBoxes[index],
                lpText: this.lpTexts[index],
                filteredText: this.filteredTexts[index],
                image: this.images[index],
                confidence: this.confidences[index],
                isTracked: this.isTracked[index]
            };
        });
    
        // Add data for the incoming license plate
        licensePlate.frames.forEach((frame, index) => {
            if (!frameToData[frame]) {
                frameToData[frame] = {
                    boundingBox: licensePlate.boundingBoxes[index],
                    lpText: licensePlate.lpTexts[index],
                    filteredText: licensePlate.filteredTexts[index],
                    image: licensePlate.images[index],
                    confidence: licensePlate.confidences[index],
                    isTracked: licensePlate.isTracked[index]
                };
            }
        });
    
        // Rebuild the arrays based on merged frames order
        this.frames = mergedFrames;
        this.boundingBoxes = mergedFrames.map(frame => frameToData[frame].boundingBox);
        this.lpTexts = mergedFrames.map(frame => frameToData[frame].lpText);
        this.filteredTexts = mergedFrames.map(frame => frameToData[frame].filteredText);
        this.images = mergedFrames.map(frame => frameToData[frame].image);
        this.confidences = mergedFrames.map(frame => frameToData[frame].confidence);
        this.isTracked = mergedFrames.map(frame => frameToData[frame].isTracked);
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
            plateItem.frame || 0,
            plateItem.is_tracked || false,
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
                plateItem.frame || 0,
                plateItem.is_tracked || false,
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
    licensePlates = licensePlates.filter((plate) => isValidPlate(plate.filteredText));
}

function joinLicensePlates() {
    // Join license plates that have a close levenshtein distance
    const threshold = 1; // Allow up to x character differences

    // licensePlates.forEach(plate => {
    //     console.log(plate.isTracked, plate.frames)
    // })

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
    // Lowercase input for case-insensitive search
    const lowerSearch = searchText.toLowerCase();

    const exactMatches = [];
    const fuzzyMatches = [];

    for (const plate of licensePlates) {
        const plateText = plate.lpText.toLowerCase();
        if (plateText.includes(lowerSearch)) {
            exactMatches.push({ plate, rank: plateText.indexOf(lowerSearch) });
        } else {
            const distance = levenshtein(plateText, lowerSearch);
            fuzzyMatches.push({ plate, distance });
        }
    }

    // Sort exact matches by where the substring appears (earlier is better)
    exactMatches.sort((a, b) => a.rank - b.rank);

    // Sort fuzzy matches by Levenshtein distance
    fuzzyMatches.sort((a, b) => a.distance - b.distance);

    // Return combined list: exact matches first, then fuzzy ones
    return [
        ...exactMatches.map(e => e.plate),
        ...fuzzyMatches.map(f => f.plate),
    ];
}