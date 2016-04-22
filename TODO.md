
# Grab-bag of todo items

## MINERvA Physics

* all: find out how to merge hdf5 files so we only have to carry around the
whole lattice in one place - plus transition to lmdb
* all: include the hcal in the lattice?
* all: generate a significant sample with FSI turned off
* vertexing
    * investigate images that go further downstream in z
    * investigate kernel padding (so we don't shrink along the z-axis after a
    convolution)
    * x/u/v single view training - test more image transforms: do they improve
    performance?
    * x/u/v transfer / initialization (single view -> all three)
    * planecode network - possible to get the plane?
    * try a target definition that includes no up/down stream plastic
* hadron multiplicity
    * set up skimmer to add final state hadron info (over some threshold)
    * maybe just charged particle multiplicity?
* EM/had fraction
* Photon/neutron separation
* Photon shower finding
* recoil energy
    * include the OD?
    * include muon info with the event?
    * hide muon digits?
* pi0 reconstruction - a la Ozgur analysis
    * set up the lattice maker to take only a subset of the digits?
    * basically - do we mask the proton and muon?
* test beam pID
* Neutral current inclusive event id (basically muon rejection?)
* separate data from mc (no selection based on event kinematics)

