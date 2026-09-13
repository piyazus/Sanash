/**
 * SANAS W2 public form router
 *
 * Deploy this project as a Web app. Visitors first choose their language;
 * then they are assigned uniformly at random to one of the four variants
 * for that language and sent directly to its Google Form.
 */

const FORMS = Object.freeze({
  RU: [
    { label: 'Block 1 — Order A', url: 'https://docs.google.com/forms/d/e/1FAIpQLSfFdxwdpMIYsddowjk4gZkj6Ef9pn8YKCZEXi5iB08V_nD8yQ/viewform' },
    { label: 'Block 1 — Order B', url: 'https://docs.google.com/forms/d/e/1FAIpQLScxnWX9TGQp61Q0SDI1klxgUBnsuJAezuh00GMoQq19OO2zAA/viewform' },
    { label: 'Block 2 — Order A', url: 'https://docs.google.com/forms/d/e/1FAIpQLSdVrTu3PmuebOYWlDrfnUvYLgSqJ-rzO2Hldkz8Aa5Xe4nnVg/viewform' },
    { label: 'Block 2 — Order B', url: 'https://docs.google.com/forms/d/e/1FAIpQLScYVsySA0FlAb4uZmUqmcIxK1evR-tZjI4LWblB-GWIKDKaeQ/viewform' },
  ],
  KK: [
    { label: 'Block 1 — Order A', url: 'https://docs.google.com/forms/d/e/1FAIpQLSebDxjwmSm-GY2gBJJNfiln0L0Pk0t1jGBDMBrIrrO7apUe5g/viewform' },
    { label: 'Block 1 — Order B', url: 'https://docs.google.com/forms/d/e/1FAIpQLSduEmf_kkOrTDQZ7Ge4HsE6YTRaB7dSeHagtIzx3Ulvubvz4g/viewform' },
    { label: 'Block 2 — Order A', url: 'https://docs.google.com/forms/d/e/1FAIpQLSdf_h7qJFJNor0PzTEU1O3KwBoJkZP3rN_AqaaqZ2AMapHpqQ/viewform' },
    { label: 'Block 2 — Order B', url: 'https://docs.google.com/forms/d/e/1FAIpQLSdmW0HwXIwnJiYU_qf9G0xDIyvuuSrxf9YlZuduMd1R5j8WWg/viewform' },
  ],
  EN: [
    { label: 'Block 1 — Order A', url: 'https://docs.google.com/forms/d/e/1FAIpQLSftxvc2Pe1a5_nAlH_UwLb5-PNnXGLDUrDrLUTK6iAp9DubyA/viewform' },
    { label: 'Block 1 — Order B', url: 'https://docs.google.com/forms/d/e/1FAIpQLScSm7Oh_L8Op4rknSlp4l9kWIcj2bgadR5Tq74ogBbgv4F0Sg/viewform' },
    { label: 'Block 2 — Order A', url: 'https://docs.google.com/forms/d/e/1FAIpQLSd7wjLwirEhg78rXJuKhahXc9mQ8MEUkqX_8v4PErZZO205FA/viewform' },
    { label: 'Block 2 — Order B', url: 'https://docs.google.com/forms/d/e/1FAIpQLSdQnTTOcQk-KK5zAn42dMjTL2LJHHd8p4M-aergytXbwQAcEA/viewform' },
  ],
});

function doGet() {
  return HtmlService.createHtmlOutputFromFile('Index')
    .setTitle('SANAS W2 Survey')
    .setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}

/** Called by the browser after the participant selects a language. */
function getRandomForm(language) {
  const key = String(language || '').trim().toUpperCase();
  if (!Object.prototype.hasOwnProperty.call(FORMS, key)) throw new Error('Unsupported language.');
  const forms = FORMS[key];
  return forms[Math.floor(Math.random() * forms.length)];
}
