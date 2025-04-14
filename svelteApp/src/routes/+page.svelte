<script>
    import CameraSettings from "../components/CameraSettings.svelte";
    import EnumSelector from "../components/EnumSelector.svelte";
    import {PiCams, cameraSettingsState} from '../state.svelte'
</script>
<div style="width:100%;height:100%">
    {#if cameraSettingsState.Loading}
    <div class="spinner-container">
        <div class="spinner"></div>
    </div>
    {/if}
    <div>
        <img class="image" src="http://localhost:8000/static/picam{cameraSettingsState.SelectedCamera}.local_snap.jpg?t={cameraSettingsState.Timestamp}" alt="Zoomable Image">
    </div>

    <CameraSettings></CameraSettings>
    <div style="position:absolute;top:10px;right:10px">
        Camera
        <EnumSelector enumToSelect={PiCams} stateField="SelectedCamera"></EnumSelector>
    </div>
</div>

<style>
    .image {
        max-width: 100%;
        height: 100%;
        object-fit: contain;
    }
    .spinner-container {
        position:absolute;
        z-index:1000;
        width:100%;
        height:100%;
        background-color:rgba(0,0,0,0.5);
    }
    .spinner {
        border: 8px solid #f3f3f3; /* Light gray */
        border-top: 8px solid #3498db; /* Blue */
        border-radius: 50%;
        width: 60px;
        height: 60px;
        animation: spin 1s linear infinite;
        position: fixed;
        top: 50%;
        left: 50%;
        margin: -30px 0 0 -30px;
        z-index: 9999;
    }
</style>