<script>
    import {cameraSettingsState} from "../state.svelte.js";
    let {enumToSelect, stateField} = $props()
    let enumNames = [];
    for (let enumMember in enumToSelect) {
        let isValueProperty = Number(enumMember) >= 0
        if (isValueProperty) {
            enumNames.push(enumToSelect[enumMember]);
        }
    }
    let selectedEnum = 0;
    let onChange = function() {
        cameraSettingsState[stateField] = selectedEnum;
    }
</script>

<select bind:value={selectedEnum} onchange={onChange}>
    {#each enumNames as enumName, i}
        <option value={i}>{enumName}</option>
    {/each}
</select>
