# ZP10D Onshape export

This configuration targets the copied Onshape assembly supplied for the ZP10D
conversion. Run it from the repository root, where `.env` provides the Onshape
API credentials:

```bash
onshape-to-robot cad/zp10d_onshape_export --save-pickle
```

The target assembly must follow `onshape-to-robot` conventions: a rigid root
assembly named `base`, revolute mates named after the 14 runtime joints, and an
`imu` mate connector/site. Export does not modify the cloud CAD document.
