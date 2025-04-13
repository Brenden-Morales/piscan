<script>
    import EnumSelector from "../components/EnumSelector.svelte";
    import RangeInput from "../components/RangeInput.svelte";
    import {AwbModes, AfTriggers, AfModes, AfRange, AfSpeed, cameraSettingsState} from "../state.svelte.js";
    let onClick = function() {
        console.log('CLICKED');
        console.log(cameraSettingsState);
        cameraSettingsState.ExposureTime += 1;
    }
    let snap = function() {
        console.log('SNAP')
    }
    let capture = function () {
        console.log('CAPTURE')
    }
</script>
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
    <RangeInput min={1} max={8} step={1}></RangeInput>
</div>
<div>
    ExposureTime:
    <input type="number" id="exposure-input" value="20000" name="number">
</div>
<div>
    <button onclick={onClick}>Settings</button>
</div>
<div>
    <button onclick={snap}>Snap</button>
</div>
<div>
    <button onclick={capture}>Capture</button>
</div>


