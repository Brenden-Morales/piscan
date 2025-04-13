<script>
    import EnumSelector from "../components/EnumSelector.svelte";
    import {AwbModes, AfTriggers, AfModes, AfRange, AfSpeed} from "../state.svelte.js";
</script>
<div>
    Awb Mode
    <EnumSelector enumToSelect={AwbModes}></EnumSelector>
</div>
<div>
    Af Mode
    <EnumSelector enumToSelect={AfModes}></EnumSelector>
</div>
<div>
    Af Trigger
    <EnumSelector enumToSelect={AfTriggers}></EnumSelector>
</div>
<div>
    Af Range
    <EnumSelector enumToSelect={AfRange}></EnumSelector>
</div>
<div>
    Af Speed
    <EnumSelector enumToSelect={AfSpeed}></EnumSelector>
</div>


