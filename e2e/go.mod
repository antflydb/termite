module github.com/antflydb/termite/e2e

go 1.26rc2

require (
	github.com/antflydb/antfly-go/libaf v0.0.0-20260119040111-36db55706c38
	github.com/antflydb/termite/pkg/client v0.0.0
	github.com/antflydb/termite/pkg/termite v0.0.0
	github.com/stretchr/testify v1.11.1
	go.uber.org/zap v1.27.1
)

require (
	github.com/ajroetker/go-highway v0.0.0-dev12 // indirect
	github.com/ajroetker/huggingface-gomlx v0.0.0-20260125071513-bbe694562131 // indirect
	github.com/apapsch/go-jsonmerge/v2 v2.0.0 // indirect
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/daulet/tokenizers v1.24.0 // indirect
	github.com/davecgh/go-spew v1.1.2-0.20180830191138-d8f796af33cc // indirect
	github.com/dlclark/regexp2 v1.11.5 // indirect
	github.com/dustin/go-humanize v1.0.1 // indirect
	github.com/eliben/go-sentencepiece v0.7.0 // indirect
	github.com/emirpasic/gods v1.18.1 // indirect
	github.com/getkin/kin-openapi v0.133.0 // indirect
	github.com/go-ini/ini v1.67.0 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/go-openapi/jsonpointer v0.22.4 // indirect
	github.com/go-openapi/swag/jsonname v0.25.4 // indirect
	github.com/gofrs/flock v0.13.0 // indirect
	github.com/gomlx/exceptions v0.0.3 // indirect
	github.com/gomlx/go-coreml v0.0.0-20260109045350-0b720b1b4a65 // indirect
	github.com/gomlx/go-coreml/gomlx v0.0.0-20260126043753-e2619e25f476 // indirect
	github.com/gomlx/go-huggingface v0.3.2-0.20260125064416-b0f56ca7fbef // indirect
	github.com/gomlx/go-xla v0.1.5-0.20260107152240-2890a4924d88 // indirect
	github.com/gomlx/gomlx v0.26.1-0.20260121051322-dbf0e3293e25 // indirect
	github.com/gomlx/onnx-gomlx v0.3.4 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/jellydator/ttlcache/v3 v3.4.0 // indirect
	github.com/josharian/intern v1.0.0 // indirect
	github.com/klauspost/compress v1.18.2 // indirect
	github.com/klauspost/cpuid/v2 v2.3.0 // indirect
	github.com/klauspost/crc32 v1.3.0 // indirect
	github.com/knights-analytics/ortgenai v0.0.3 // indirect
	github.com/mailru/easyjson v0.9.1 // indirect
	github.com/minio/crc64nvme v1.1.1 // indirect
	github.com/minio/md5-simd v1.1.2 // indirect
	github.com/minio/minio-go/v7 v7.0.97 // indirect
	github.com/mitchellh/colorstring v0.0.0-20190213212951-d06e56a500db // indirect
	github.com/mohae/deepcopy v0.0.0-20170929034955-c48cc78d4826 // indirect
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822 // indirect
	github.com/oapi-codegen/runtime v1.1.2 // indirect
	github.com/oasdiff/yaml v0.0.0-20250309154309-f31be36b4037 // indirect
	github.com/oasdiff/yaml3 v0.0.0-20250309153720-d2182401db90 // indirect
	github.com/perimeterx/marshmallow v1.1.5 // indirect
	github.com/philhofer/fwd v1.2.0 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/pkoukk/tiktoken-go v0.1.8 // indirect
	github.com/pkoukk/tiktoken-go-loader v0.0.2 // indirect
	github.com/pmezard/go-difflib v1.0.1-0.20181226105442-5d4384ee4fb2 // indirect
	github.com/prometheus/client_golang v1.23.2 // indirect
	github.com/prometheus/client_model v0.6.2 // indirect
	github.com/prometheus/common v0.67.5 // indirect
	github.com/prometheus/procfs v0.19.2 // indirect
	github.com/rivo/uniseg v0.4.7 // indirect
	github.com/rs/xid v1.6.0 // indirect
	github.com/schollz/progressbar/v2 v2.15.0 // indirect
	github.com/sugarme/regexpset v0.0.0-20200920021344-4d4ec8eaf93c // indirect
	github.com/sugarme/tokenizer v0.3.0 // indirect
	github.com/tinylib/msgp v1.6.3 // indirect
	github.com/woodsbury/decimal128 v1.4.0 // indirect
	github.com/x448/float16 v0.8.4 // indirect
	github.com/yalue/onnxruntime_go v1.25.0 // indirect
	go.uber.org/multierr v1.11.0 // indirect
	go.yaml.in/yaml/v2 v2.4.3 // indirect
	golang.org/x/crypto v0.47.0 // indirect
	golang.org/x/exp v0.0.0-20260112195511-716be5621a96 // indirect
	golang.org/x/image v0.34.0 // indirect
	golang.org/x/net v0.49.0 // indirect
	golang.org/x/sync v0.19.0 // indirect
	golang.org/x/sys v0.40.0 // indirect
	golang.org/x/term v0.39.0 // indirect
	golang.org/x/text v0.33.0 // indirect
	google.golang.org/protobuf v1.36.11 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
	k8s.io/klog/v2 v2.130.1 // indirect
)

replace github.com/antflydb/termite/pkg/client => ../pkg/client

replace github.com/antflydb/termite/pkg/termite => ../pkg/termite

replace github.com/gomlx/gomlx => github.com/timkaye11/gomlx v0.0.0-20251221044952-ee84bc4bbaf7

replace github.com/gomlx/onnx-gomlx => github.com/timkaye11/onnx-gomlx v0.0.0-20251217194823-a871bb8cc687
