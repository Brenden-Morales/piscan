export class CameraSettings {
    /**
     *
     * @param {HTMLElement} awbContainer
     * @param {HTMLElement} exposureContainer
     * @param {HTMLButtonElement} settingsButton
     */
    constructor(awbContainer, exposureContainer, settingsButton) {
        this.awbContainer = awbContainer
        this.awb = awbContainer.querySelector('#awb-mode')
        this.awbValue = awbContainer.querySelector('#awb-value')
        this.awbRangeChanged = this.awbRangeChanged.bind(this)
        this.awb.addEventListener('input',this.awbRangeChanged)

        this.exposureContainer = exposureContainer
        this.exposureInput = exposureContainer.querySelector('#exposure-input')

        this.settingsButton = settingsButton
        this.settingsButtonClicked = this.settingsButtonClicked.bind(this)
        this.settingsButton.addEventListener('click', this.settingsButtonClicked)
    }

    /**
     *
     * @param {InputEvent}event
     */
    awbRangeChanged(event) {
        this.awbValue.textContent = event.target.value
    }

    /**
     *
     */
    settingsButtonClicked() {
        fetch("/api/settings", {
            method: "PUT", // or "POST"
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                AwbMode: this.awb.value,
                ExposureTime: this.exposureInput.value
            })
        })
        .then(response => response.json())
        .then(result => {
            console.log("Success:", result);
        })
        .catch(error => {
            console.error("Error:", error);
        });
    }

}