<script>
    let {min, max, step} = $props()
    let value = $state(min);
</script>

<div>
    <input type="range" name="number" min={min} max={max} bind:value={value} step={step}>
    <span>{value}</span>
</div>
