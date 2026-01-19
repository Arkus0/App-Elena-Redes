export class DomLayoutChangedError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'DomLayoutChangedError';
    Object.setPrototypeOf(this, DomLayoutChangedError.prototype);
  }
}
