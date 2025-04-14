<script>
    import EnumSelector from "./EnumSelector.svelte";
    import {AwbModes, AfTriggers, AfModes, AfRange, AfSpeed, cameraSettingsState} from '../state.svelte'
    let onClick = function() {
        console.log('SETTINGS');
        console.log(cameraSettingsState);
    }
    let snap = function() {
        console.log('SNAP')
    }
    let capture = function () {
        console.log('CAPTURE')
    }
</script>
<div style="position:relative;top:0px;left:0px">
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
        <button onclick={onClick}>Apply Settings</button>
    </div>
    <div>
        <button onclick={snap}>Snap</button>
    </div>
    <div>
        <button onclick={capture}>Capture</button>
    </div>

</div>