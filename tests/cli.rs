use anyhow::Result;
use assert_cmd::Command;
use predicates::prelude::*;

#[test]
#[ignore]
fn dies_no_args() -> Result<()> {
    let mut cmd = Command::cargo_bin("haplmate")?;
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("error"));
    Ok(())
}
