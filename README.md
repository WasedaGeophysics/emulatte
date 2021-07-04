# Emulatte 1.0.0 Pre-Release Note
## Emulatte : EM-exploratory simulation & analysis software
<p> Emulatte is the modern software for geophysical electromagnetic exploration. Emulatte supports homogeneous and horizontal multi-layered structures as envisioned underground structural models.
</p>

- Documentation
- Contact
- Bag Report

## Installation
- Requirements : numpy and scipy 

 In your terminal, command
<code>

    ~ % pip install emulatte

</code>

## Usage

<code>

    import emulatte.forward as fwd

    model = fwd.model(thicks)
    vmd = fwd.transceiver('VMD', freqtime, dipole_moment=1)
    model.add_resistivity(res)
    model.add_permeability(res)

    ...

    model.locate(vmd, tc, rc)
    ans, _ = model.emulate()

</code>


<p>Geophysics Lab. @ Waseda University</p>
<p>早稲田大学 物理探査工学研究室</p>