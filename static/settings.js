export class CameraSettings {
    /**
     *
     * @param {HTMLElement} awbContainer
     * @param {HTMLElement} exposureContainer
     * @param {HTMLElement} analogGainContainer
     * @param {HTMLElement} afModeContainer
     * @param {HTMLElement} afTriggerContainer
     * @param {HTMLButtonElement} settingsButton
     */
    constructor(awbContainer, exposureContainer, analogGainContainer, afModeContainer, afTriggerContainer, settingsButton) {
        this.awbContainer = awbContainer
        this.awb = awbContainer.querySelector('#awb-mode')
        this.awbValue = awbContainer.querySelector('#awb-value')
        this.awbRangeChanged = this.awbRangeChanged.bind(this)
        this.awb.addEventListener('input',this.awbRangeChanged)

        this.analogGainContainer = analogGainContainer
        this.analogGain = analogGainContainer.querySelector('#analog-gain')
        this.analogGainValue = analogGainContainer.querySelector('#analog-gain-value')
        this.analogGainChanged = this.analogGainChanged.bind(this)
        this.analogGain.addEventListener('input',this.analogGainChanged)

        this.afModeContainer = afModeContainer
        this.afMode = afModeContainer.querySelector('#af-mode')
        this.afModeValue = afModeContainer.querySelector('#af-mode-value')
        this.afModeChanged = this.afModeChanged.bind(this)
        this.afMode.addEventListener('input',this.afModeChanged)

        this.afTriggerContainer = afTriggerContainer
        this.afTrigger = afTriggerContainer.querySelector('#af-trigger')
        this.afTriggerValue = afTriggerContainer.querySelector('#af-trigger-value')
        this.afTriggerChanged = this.afTriggerChanged.bind(this)
        this.afTrigger.addEventListener('input',this.afTriggerChanged)

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
     * @param {InputEvent}event
     */
    analogGainChanged(event) {
        this.analogGainValue.textContent = event.target.value
    }

    /**
     *
     * @param {InputEvent}event
     */
    afModeChanged(event) {
        this.afModeValue.textContent = event.target.value
    }

    /**
     *
     * @param {InputEvent}event
     */
    afTriggerChanged(event) {
        this.afTriggerValue.textContent = event.target.value
    }

    /**
     *
     */
    settingsButtonClicked() {
        console.log(this.afTrigger.value)
        fetch("/api/settings", {
            method: "PUT", // or "POST"
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                AwbMode: this.awb.value,
                ExposureTime: this.exposureInput.value,
                AnalogueGain: this.analogGain.value,
                AfMode: this.afMode.value,
                AfTrigger: this.afTrigger.value,
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