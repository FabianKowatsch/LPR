
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

function isValidPlate(plateText) {
    if (plateText.length < 3 || plateText.length > 9) return false; // Length check
    if (/^[0-9]+$/.test(plateText)) return false; // Only numbers (e.g., "11", "000")
    if (/^[A-Z]+$/.test(plateText)) return false; // Only letters (e.g., "VIII", "WIN")
    // if (/[^A-Z0-9-_]/.test(plateText)) return false; // Contains special characters (e.g., "#", "..."), "-" and "_" allowed

    return true;
}

// Function to select an example image
function selectExample(element) {
    const fileInput = document.getElementById("fileInput");
    // Falls bereits eine Datei hochgeladen wurde, frage den Nutzer
    if (fileInput.files.length > 0) {
        const useExample = confirm(
            "A file has already been uploaded. Do you want to remove the file and select the example instead?"
        );
        if (!useExample) {
            return; // Wenn der Nutzer ablehnt, nichts ändern
        }
        // Datei entfernen
        fileInput.value = "";
    }
    const exampleSelect = document.getElementById("exampleSelect");
    const exampleImages = document.querySelectorAll(".example-img");

    // Set the hidden input value to the selected example's filename
    exampleSelect.value = element.getAttribute("data-filename");

    // Highlight the selected image
    exampleImages.forEach((img) => img.classList.remove("selected"));
    element.classList.add("selected");
}

function toggleSwitchLabel() {
    const label = document.getElementById("toggleLabel");
    label.innerText = toggleBoxes.checked ? "Show Tracked Boxes" : "Show Raw Boxes";
}