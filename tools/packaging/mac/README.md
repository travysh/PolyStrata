# Signing and notarising the Mac builds

`sign-and-notarize.sh` signs the Mac Terminal archive and Mac engine packages with the Developer ID Application certificate, sends them
to Apple's notary service and checks the result. After it, macOS runs them without the "damaged" / "unidentified developer" warning.

It has to run on the Mac that holds the certificate. It never asks for, reads or stores a password or a key: `codesign` and
`notarytool` take them from the Keychain. The only names in it are the certificate (`Developer ID Application: Niko Veit (79LR4R7RSC)`)
and the notarytool profile (`strata-notary`); `SIGN_ID` and `NOTARY_PROFILE` override them.

## What to type (once per build)

1. Put these on the Mac in one folder: `sign-and-notarize.sh`, `entitlements-node.plist` (they must stay together), and the file(s) to sign,
   for example `Strata-Terminal-Mac.zip`.
2. Open Terminal and go to that folder:  `cd ~/Downloads/sign`
3. Check the two things the script needs (both should answer without an error):

       security find-identity -v -p codesigning        # lists "Developer ID Application: Niko Veit (79LR4R7RSC)"
       xcrun notarytool history --keychain-profile strata-notary

4. Run it:

       bash sign-and-notarize.sh --terminal Strata-Terminal-Mac.zip

   macOS may ask once whether `codesign` may use the key in your Keychain: type your Mac login password and choose "Always Allow".
   Then wait: Apple's check takes a few minutes per file (up to about 15 minutes each; the script waits and prints when it is done).
5. Read the summary at the end. Every line is `OK`; the finished files are in the `signed` folder next to the input, with their sha256
   (also in `signed/SHA256SUMS.txt`) and the full log (`signed/sign-and-notarize.log`).
6. If a line says `FAIL`, the run stopped there. If Apple rejected a file, the script prints Apple's log and saves it as
   `signed/notary-log-*.json`. Send us that file (it contains file names and reasons, no secrets).

A trial run that signs ad-hoc and sends nothing to Apple: add `--dry-run` (no certificate needed).

## An engine package from the update feed

An engine package from the update feed (the engine of a model shape: its program with its `metal/`
sources and `engine-component.json`) is signed here as its own input: `--engine <component>-<version>-macos-arm64-metal-unsigned-folder.tar.gz`, alone or
together with `--terminal` in one run. The script signs every program in the folder with the same identity and options
(hardened runtime, timestamp, no entitlements), writes the new SHA-256 and size into `engine-component.json`,
runs `codesign --verify --strict`, starts the signed program once (no model), sends a zip of the folder to Apple and waits, and
writes `signed/<component>-<version>-macos-arm64-metal-folder.tar.gz` (also listed in `signed/SHA256SUMS.txt`). On the PC, sign that
folder with the feed key (this adds the release signature to the manifest) and build the feed from it; the unsigned folder must not be published.
`--dry-run` signs ad-hoc and skips Apple, as for everything else.

## What it does

For the Terminal archive it signs every Mach-O inside (hardened runtime and a secure timestamp), packs it again in the format it came
in (`.zip` or `.tar.gz`) and notarises it. For an engine package it does what the section above says. A finished file is written to
`signed/` only after every step for it has passed. The steps are listed at the top of the script.

## Entitlements

The Terminal archive and the engine packages get none: Python, the engine and all libraries are signed with the hardened runtime and
no entitlement. `disable-library-validation` is not used: library validation only refuses a library signed by somebody else, and
every library in these bundles is re-signed with the one identity.

The engine components Control downloads later (`strata update`) are unpacked without the download mark and run as their own
processes, so unsigned components keep working. Metal compiles the engine's kernel sources through the system compiler service,
which was tested under the hardened runtime.

## The Terminal zip

Apple cannot staple a ticket to bare command-line programs in a zip. The zip is still notarised (Apple records the programs by hash),
and macOS looks that up online the first time a browser-downloaded copy is opened. Installs done with `curl ... | sh` are not
quarantined at all, so this mainly helps zips downloaded in a browser.

Signing changes the bytes of the zip and of every program inside, so:

* a checksum list or signed feed (`strata-feed.json`, `*.manifest.json`) that names the old zip or archive must be rebuilt from the new one;
* an archive that holds a *feed* (the tester Terminal zip: scripts plus `feed/*.tar.gz`) has no program at its top level. By default
  the nested `.tar.gz` files are left alone (they are unpacked by command-line tools and carry no download mark). `--sign-nested`
  signs inside them and packs them again, but then the feed's `sha256` entries and its signature (`strata-feed.json.sig`, made with the
  release key) no longer match and the feed has to be regenerated and re-signed.
