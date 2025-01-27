library(stats)
fourier_transform <- function(data) {
  spectrum <- fft(data)
  return(Mod(spectrum))
}
library(wavelets)
wavelet_transform <- function(data) {
  wt <- dwt(data, filter = "haar", n.levels = 4)
  return(unlist(wt@W))
}
walsh_transform <- function(data) {
  walsh_mat <- function(n) {
    if (n == 1) return(matrix(c(1, 1, 1, -1), 2, 2))
    w <- walsh_mat(n - 1)
    return(kronecker(matrix(c(1, 1, 1, -1), 2, 2), w))
  }
  walsh <- walsh_mat(log2(length(data)))
  return(data %*% walsh)
}
library(pracma)
hilbert_huang_transform <- function(data) {
  emd_res <- emd(data)
  imfs <- emd_res$residue
  hilbert_res <- Hilbert(imfs)
  return(Mod(hilbert_res))
}