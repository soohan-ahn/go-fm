package fm

import (
	"hash/adler32"
)

func (fm *FM) FeatureHash(f string) uint32 {
	return (adler32.Checksum([]byte(f))) % uint32(*fm.Params.FM.MaxDimension)
}

func (fm *FM) ReadLine(line []string) []uint32 {
	feature := []uint32{}

	// Implement by your own.
	// TODO: Make an interface related to this method.
	for i, f := range line {
		if i == 0 || i == 1 || i == 3 || i == 6 || i == 9 || i == 10 {
			if i == 9 && f == " " {
				f = "0"
			} else if i == 9 {
				f = "1"
			}
			featureNum := FeatureHash(fm, f)
			feature = append(feature, featureNum)
		}
	}
	return feature
}
