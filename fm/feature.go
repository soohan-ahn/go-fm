package fm

type FeatureExtractor interface {
	ReadLine([]string) []uint32
	FeatureHash(string) uint32
}

func ReadLine(fe FeatureExtractor, line []string) []uint32 {
	return fe.ReadLine(line)
}

func FeatureHash(fe FeatureExtractor, f string) uint32 {
	return fe.FeatureHash(f)
}
