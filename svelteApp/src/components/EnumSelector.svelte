<script>
    let {enumToSelect} = $props()
    let enumNames = [];
    for (let enumMember in enumToSelect) {
        let isValueProperty = Number(enumMember) >= 0
        if (isValueProperty) {
            enumNames.push(enumToSelect[enumMember]);
        }
    }
    let selectedEnum = 0;
</script>

<select bind:value={selectedEnum}>
    {#each enumNames as enumName, i}
        <option value={i}>{enumName}</option>
    {/each}
</select>
