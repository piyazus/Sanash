// Read-only availability check of the 12 already configured public forms.
// Never submits responses; HTTP success does not prove a working submission.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '../../..');
const router = vm.createContext({});
vm.runInContext(fs.readFileSync(path.join(root, 'sanas_form_router/Code.gs'), 'utf8'), router);
const forms = vm.runInContext('FORMS', router);
(async () => {
  const results = await Promise.all(Object.entries(forms).flatMap(([language, variants]) => variants.map(async form => {
    const row = { language, variant: form.label, url: form.url, checked_at_utc: new Date().toISOString() };
    try {
      const response = await fetch(form.url, { signal: AbortSignal.timeout(20000) });
      const html = await response.text();
      return { ...row, http_status: response.status, final_url: response.url,
        title: (html.match(/<title>([^<]*)<\/title>/i) || [])[1] || '',
        form_payload_present: html.includes('FB_PUBLIC_LOAD_DATA_'),
        availability: response.ok && html.includes('FB_PUBLIC_LOAD_DATA_') ? 'FORM_PAGE_RECEIVED' : 'REVIEW_REQUIRED' };
    } catch (error) { return { ...row, availability: 'FETCH_FAILED', error: error.message }; }
  })));
  const output = path.join(root, 'deliverables/progress_2026-09-12');
  fs.mkdirSync(output, { recursive: true });
  fs.writeFileSync(path.join(output, 'public_forms_check.json'), JSON.stringify({
    scope: 'Public GET only; submission, acceptance of responses, source mapping and master trigger are not verified.', results
  }, null, 2) + '\n');
  for (const row of results) console.log(`${row.language} ${row.variant}: ${row.availability} (${row.http_status || row.error})`);
  if (results.some(row => row.availability !== 'FORM_PAGE_RECEIVED')) process.exitCode = 1;
})();
