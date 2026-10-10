# Learned object selection

{
  "status": "COMPLETE_NO_TARGET_GAIN",
  "flags": {
    "LR02_FC_2": {
      "all5_checks": {
        "replica8": {
          "apall": false,
          "ap50": false,
          "ap25": true,
          "miou": true,
          "macc": false
        },
        "scannet_cf18": {
          "apall": true,
          "ap50": true,
          "ap25": true,
          "miou": true,
          "macc": true
        }
      },
      "CF_D2_APall_strict": false,
      "CF_D2_AP50_nondecrease": true,
      "target_met": false,
      "material_target_met": false
    },
    "LR03_FC_4": {
      "all5_checks": {
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
        }
      },
      "CF_D2_APall_strict": true,
      "CF_D2_AP50_nondecrease": true,
      "target_met": false,
      "material_target_met": false
    },
    "LR04_FC_8": {
      "all5_checks": {
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
        }
      },
      "CF_D2_APall_strict": false,
      "CF_D2_AP50_nondecrease": false,
      "target_met": false,
      "material_target_met": false
    },
    "LR05_MA_8": {
      "all5_checks": {
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
          "ap25": false,
          "miou": false,
          "macc": false
        }
      },
      "CF_D2_APall_strict": false,
      "CF_D2_AP50_nondecrease": false,
      "target_met": false,
      "material_target_met": false
    },
    "LR06_MV_VIEW": {
      "all5_checks": {
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
          "ap25": false,
          "miou": false,
          "macc": false
        }
      },
      "CF_D2_APall_strict": false,
      "CF_D2_AP50_nondecrease": false,
      "target_met": false,
      "material_target_met": false
    },
    "LR07_MV_SURFACE": {
      "all5_checks": {
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
          "ap25": false,
          "miou": false,
          "macc": false
        }
      },
      "CF_D2_APall_strict": false,
      "CF_D2_AP50_nondecrease": false,
      "target_met": false,
      "material_target_met": false
    },
    "LR08_MV_AUX": {
      "all5_checks": {
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
          "ap25": false,
          "miou": false,
          "macc": false
        }
      },
      "CF_D2_APall_strict": false,
      "CF_D2_AP50_nondecrease": false,
      "target_met": false,
      "material_target_met": false
    }
  },
  "passing_candidates": [],
  "best_regression_candidate": null,
  "research_retained": "LR01_G1",
  "target_met": false,
  "material_target_met": false,
  "selection_scope": "EXPLORATORY_ON_REPEATEDLY_EXPOSED_BENCHMARKS",
  "runtime_in_gate": false,
  "deployment": "N0_UNCHANGED",
  "dev_nomination": "f0e9df5a41e2142cea5cad6d697bb399932968ac4666e8c7d42e58aa1496e56a",
  "dev_proposed_architecture": "LR08_MV_AUX",
  "repeat_nomination": "8c560f7f7bc728e913a3efc8f770f599850953918e89d56adcb34afd804a4f62",
  "repeat_consistency": {
    "flags": {
      "LR05_MA_8": {
        "all5_checks": {
          "replica8": {
            "apall": true,
            "ap50": false,
            "ap25": false,
            "miou": false,
            "macc": false
          },
          "scannet_cf18": {
            "apall": false,
            "ap50": false,
            "ap25": false,
            "miou": false,
            "macc": false
          }
        },
        "CF_D2_APall_strict": false,
        "CF_D2_AP50_nondecrease": false,
        "target_met": false,
        "material_target_met": false
      },
      "LR08_MV_AUX": {
        "all5_checks": {
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
            "ap25": false,
            "miou": false,
            "macc": false
          }
        },
        "CF_D2_APall_strict": false,
        "CF_D2_AP50_nondecrease": false,
        "target_met": false,
        "material_target_met": false
      }
    },
    "seed29_minus_seed17": {
      "replica8": {
        "LR05_MA_8": {
          "apall": 0.012594594790236988,
          "ap50": 0.02730582314086369,
          "ap25": 0.014905725836113215,
          "miou": 0.0013120657824661208,
          "macc": 0.0120590138099258
        },
        "LR08_MV_AUX": {
          "apall": 0.005269002475222814,
          "ap50": 0.02054745984413045,
          "ap25": 0.04271788987563019,
          "miou": 0.010205183477031643,
          "macc": 0.01592655133104859
        }
      },
      "scannet_cf18": {
        "LR05_MA_8": {
          "apall": 0.0059241505312038895,
          "ap50": 0.010882665053944085,
          "ap25": 0.01232064216971357,
          "miou": 0.012411621315918775,
          "macc": 0.01605316281645025
        },
        "LR08_MV_AUX": {
          "apall": 0.0030690205961770717,
          "ap50": 0.0023962440698523657,
          "ap25": -0.0014329572688739634,
          "miou": -0.001582575721070456,
          "macc": 0.0025740084055483337
        }
      }
    },
    "proposed_minus_MA_by_seed": {
      "replica8": {
        "seed17": {
          "apall": -0.003784059770056261,
          "ap50": -0.0005843407083452323,
          "ap25": -0.02003752424086863,
          "miou": -0.0022428321512679705,
          "macc": -0.0004001199840806824
        },
        "seed29": {
          "apall": -0.011109652085070434,
          "ap50": -0.007342704005078471,
          "ap25": 0.007774639798648342,
          "miou": 0.006650285543297552,
          "macc": 0.0034674175370421056
        }
      },
      "scannet_cf18": {
        "seed17": {
          "apall": 0.006822503217842155,
          "ap50": 0.016364536350568365,
          "ap25": 0.023364804306798687,
          "miou": 0.018882081413532176,
          "macc": 0.024081226725130578
        },
        "seed29": {
          "apall": 0.003967373282815337,
          "ap50": 0.007878115366476646,
          "ap25": 0.009611204868211154,
          "miou": 0.004887884376542945,
          "macc": 0.010602072314228661
        }
      }
    }
  },
  "research_retained_replication": "UNTRAINED_CONTROL",
  "parameter_counts": {
    "LR05_MA_8": 1829540,
    "LR06_MV_VIEW": 2127781,
    "LR07_MV_SURFACE": 2127781,
    "LR08_MV_AUX": 2127781
  },
  "independent_confirmation": false,
  "training_or_seed_selection_uses_regression": false,
  "identity": "19a4875d55adca5b4c6920f95e36b07815babd4c1cff4a4e01f9d624c8472db6"
}

DEV checkpoint/architecture selection precedes H. The regression winner is exploratory on exposed benchmarks. A main learned winner outside the pre-nominated repeat pair is a SINGLE_SEED_CANDIDATE. Deployment remains N0_UNCHANGED.
