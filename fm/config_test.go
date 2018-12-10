package fm_test

import (
	"testing"

	. "github.com/soohanboys/go-fm/fm"
)

func TestReadParmas(t *testing.T) {
	cfg, err := ReadParams()
	if err != nil {
		t.Fatalf("Unexpected Error: %v\n", err)
	}

	if cfg.FM == nil {
		t.Fatalf("FM should not be nil.\n")
	}
	if d := cfg.FM.Degree; d == nil || *d != 2 {
		t.Fatalf("degree should be 2 not %v.\n", d)
	}

	if cfg.Files == nil {
		t.Fatalf("Files should not be nil.\n")
	}
	if p := cfg.Files.PredictFileName; p == nil || *p != "for_validation/test.csv" {
		t.Fatalf("PredictFileName should be %v not %v.\n", "for_validation/test.csv", p)
	}
}
