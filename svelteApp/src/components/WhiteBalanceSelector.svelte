<script>
    import {AwbModes} from "../state.svelte.js";
    let ModeNames = [];
    for (let enumMember in AwbModes) {
        let isValueProperty = Number(enumMember) >= 0
        if (isValueProperty) {
            ModeNames.push(AwbModes[enumMember]);
        }
    }
    let selectedMode = 0;

</script>

<select bind:value={selectedMode}>
    {#each ModeNames as modeName, i}
        <option value={i}>{modeName}</option>
    {/each}
</select>
