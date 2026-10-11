# Frozen FC selection

{
  "DEV_NOMINEE": "FG2_SURFACE",
  "EXPOSED_BENCHMARK_CANDIDATE": null,
  "benchmark_selection_exploratory": true,
  "research_retained": "REF_G1",
  "deployment": "N0_UNCHANGED",
  "gates": {
    "FG1_VIEW": {
      "passed": false,
      "checks": {
        "replica8": {
          "apall": false,
          "ap50": false,
          "ap25": true,
          "miou": false,
          "macc": false
        },
        "scannet_cf18": {
          "apall": false,
          "ap50": true,
          "ap25": true,
          "miou": true,
          "macc": true
        },
        "CF_vs_D2": {
          "APall_gt": false,
          "AP50_ge": true
        }
      },
      "CF_APall_gain_fraction": -0.001508933548497432,
      "material_CF_gain": false
    },
    "FG2_SURFACE": {
      "passed": false,
      "checks": {
        "replica8": {
          "apall": false,
          "ap50": true,
          "ap25": true,
          "miou": true,
          "macc": true
        },
        "scannet_cf18": {
          "apall": false,
          "ap50": true,
          "ap25": true,
          "miou": true,
          "macc": true
        },
        "CF_vs_D2": {
          "APall_gt": false,
          "AP50_ge": false
        }
      },
      "CF_APall_gain_fraction": -0.0012163131182456044,
      "material_CF_gain": false
    },
    "FG3_AUX": {
      "passed": false,
      "checks": {
        "replica8": {
          "apall": false,
          "ap50": true,
          "ap25": true,
          "miou": true,
          "macc": true
        },
        "scannet_cf18": {
          "apall": false,
          "ap50": false,
          "ap25": true,
          "miou": true,
          "macc": true
        },
        "CF_vs_D2": {
          "APall_gt": false,
          "AP50_ge": false
        }
      },
      "CF_APall_gain_fraction": -0.000887060831628983,
      "material_CF_gain": false
    },
    "FG4_ROUTE": {
      "passed": false,
      "checks": {
        "replica8": {
          "apall": false,
          "ap50": false,
          "ap25": true,
          "miou": true,
          "macc": true
        },
        "scannet_cf18": {
          "apall": false,
          "ap50": true,
          "ap25": true,
          "miou": true,
          "macc": true
        },
        "CF_vs_D2": {
          "APall_gt": false,
          "AP50_ge": false
        }
      },
      "CF_APall_gain_fraction": -0.001002079202586048,
      "material_CF_gain": false
    },
    "S29_FG1_VIEW": {
      "passed": false,
      "checks": {
        "replica8": {
          "apall": false,
          "ap50": false,
          "ap25": false,
          "miou": false,
          "macc": false
        },
        "scannet_cf18": {
          "apall": false,
          "ap50": false,
          "ap25": true,
          "miou": true,
          "macc": true
        },
        "CF_vs_D2": {
          "APall_gt": false,
          "AP50_ge": false
        }
      },
      "CF_APall_gain_fraction": -0.0022311313520523013,
      "material_CF_gain": false
    },
    "S29_FG2_SURFACE": {
      "passed": false,
      "checks": {
        "replica8": {
          "apall": false,
          "ap50": false,
          "ap25": true,
          "miou": true,
          "macc": true
        },
        "scannet_cf18": {
          "apall": true,
          "ap50": true,
          "ap25": true,
          "miou": true,
          "macc": true
        },
        "CF_vs_D2": {
          "APall_gt": false,
          "AP50_ge": true
        }
      },
      "CF_APall_gain_fraction": -0.0004589993563415551,
      "material_CF_gain": false
    },
    "S29_FG3_AUX": {
      "passed": false,
      "checks": {
        "replica8": {
          "apall": false,
          "ap50": true,
          "ap25": true,
          "miou": true,
          "macc": true
        },
        "scannet_cf18": {
          "apall": false,
          "ap50": true,
          "ap25": true,
          "miou": true,
          "macc": true
        },
        "CF_vs_D2": {
          "APall_gt": false,
          "AP50_ge": false
        }
      },
      "CF_APall_gain_fraction": -0.0017208043204112178,
      "material_CF_gain": false
    },
    "S29_FG4_ROUTE": {
      "passed": false,
      "checks": {
        "replica8": {
          "apall": false,
          "ap50": false,
          "ap25": true,
          "miou": true,
          "macc": true
        },
        "scannet_cf18": {
          "apall": false,
          "ap50": true,
          "ap25": true,
          "miou": true,
          "macc": true
        },
        "CF_vs_D2": {
          "APall_gt": false,
          "AP50_ge": false
        }
      },
      "CF_APall_gain_fraction": -0.0017369057980359537,
      "material_CF_gain": false
    }
  },
  "latency_gate": false,
  "status": "COMPLETE_NO_TARGET_GAIN",
  "same_architecture_repeat": null,
  "repeat_map_pass": null,
  "scientific_status": "COMPLETE_NO_TARGET_GAIN",
  "identity": "5a424467208020cdfdba60cab9a3b6c2b762f6ad677b341a3551e0d8567f81d9"
}

A/M/C use69 DEV base originals with full200 competitors. Qualification requires A gain>=0.005, nondecreasing M and C, and positive raw net corrections. Checkpoints250/500/1000/1500/2000 rank by A/M/C descending, base160 CE ascending, then earlier step. Step0 cannot qualify. Same-step contrasts and independently selected-step contrasts are separate. Seed29 cannot replace the seed17 architecture. Maps/H2/runtime never choose checkpoints.
