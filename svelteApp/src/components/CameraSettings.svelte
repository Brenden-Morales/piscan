<script>
    import EnumSelector from "./EnumSelector.svelte";
    import {AwbModes, AfTriggers, AfModes, AfRange, AfSpeed, cameraSettingsState} from '../state.svelte'
    let onSettings = function() {
        console.log('SETTINGS');
        console.log(cameraSettingsState);
        cameraSettingsState.Loading = true;
        fetch("http://localhost:8000/api/settings", {
            method: "PUT", // or "POST"
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                AwbMode: cameraSettingsState.AwbMode,
                AfMode: cameraSettingsState.AfMode,
                AfTrigger: cameraSettingsState.AfTrigger,
                AfRange: cameraSettingsState.AfRange,
                AfSpeed: cameraSettingsState.AfSpeed,
                AnalogueGain: cameraSettingsState.AnalogueGain,
                ExposureTime: cameraSettingsState.ExposureTime
            })
        })
            .then(response => response.json())
            .then(result => {
                console.log("Success:", result);
                cameraSettingsState.Loading = false;
            })
            .catch(error => {
                console.error("Error:", error);
                cameraSettingsState.Loading = false;
            });
    }
    let snapAllCams = function(url) {
        cameraSettingsState.Loading = true;
        fetch(url, {
            method: "PUT", // or "POST"
            headers: {
                "Content-Type": "application/json"
            }
        })
            .then(response => response.json())
            .then(result => {
                console.log("Success:", result);
                for(let i = 0; i < 6; i ++) {
                    cameraSettingsState.Timestamp = Date.now()
                    cameraSettingsState.Loading = false;
                }
            })
            .catch(error => {
                console.error("Error:", error);
                cameraSettingsState.Loading = false;
            });
    }
    let snap = function () {
        snapAllCams("http://localhost:8000/api/snap")
    }
    let capture = function () {
        snapAllCams("http://localhost:8000/api/capture")
    }
</script>
<div style="position:absolute;top:0px;left:0px">
    <div>
        Awb Mode
        <EnumSelector enumToSelect={AwbModes} stateField="AwbMode"></EnumSelector>
    </div>
    <div>
        Af Mode
        <EnumSelector enumToSelect={AfModes} stateField="AfMode"></EnumSelector>
    </div>
    <div>
        Af Trigger
        <EnumSelector enumToSelect={AfTriggers} stateField="AfTrigger"></EnumSelector>
    </div>
    <div>
        Af Range
        <EnumSelector enumToSelect={AfRange} stateField="AfRange"></EnumSelector>
    </div>
    <div>
        Af Speed
        <EnumSelector enumToSelect={AfSpeed} stateField="AfSpeed"></EnumSelector>
    </div>
    <div style="display:flex">
        <div>Analog Gain</div>
        <input type="range" name="number" min=1 max=8 bind:value={cameraSettingsState.AnalogueGain} step=1>
        <span>{cameraSettingsState.AnalogueGain}</span>
    </div>
    <div>
        ExposureTime:
        <input type="number" id="exposure-input" bind:value={cameraSettingsState.ExposureTime} name="number">
    </div>
    <div>
        <button onclick={onSettings}>Apply Settings</button>
    </div>
    <div>
        <button onclick={snap}>Snap</button>
    </div>
    <div>
        <button onclick={capture}>Capture</button>
    </div>

</div>