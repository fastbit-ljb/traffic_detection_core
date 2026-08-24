import { Check, ChevronDown } from 'lucide-react';
import {
  Button,
  Header,
  ListBox,
  ListBoxItem,
  ListBoxSection,
  Popover,
  Select as SelectPrimitive,
  SelectValue as SelectValuePrimitive,
  type SelectProps,
} from 'react-aria-components';
import type { ReactNode } from 'react';

function mergeClassName(...classNames: Array<string | undefined>) {
  return classNames.filter(Boolean).join(' ');
}

type SimpleProps = {
  className?: string;
  children: ReactNode;
};

export function Select<T extends object>({ className, ...props }: Omit<SelectProps<T>, 'className'> & { className?: string }) {
  return <SelectPrimitive {...props} className={mergeClassName('app-select', className)} />;
}

export function SelectTrigger({ className, children }: SimpleProps) {
  return <Button className={mergeClassName('app-select-trigger', className)}>{children}<ChevronDown className="app-select-chevron" size={16} aria-hidden="true" /></Button>;
}

export function SelectValue({ className }: { className?: string }) {
  return <SelectValuePrimitive className={mergeClassName('app-select-value', className)} />;
}

export function SelectContent({ className, children }: SimpleProps) {
  return <Popover className={mergeClassName('app-select-popover', className)}><ListBox className="app-select-listbox">{children}</ListBox></Popover>;
}

export function SelectGroup({ className, children }: SimpleProps) {
  return <ListBoxSection className={mergeClassName('app-select-group', className)}>{children}</ListBoxSection>;
}

export function SelectLabel({ className, children }: { className?: string; children: ReactNode }) {
  return <Header className={mergeClassName('app-select-label', className)}>{children}</Header>;
}

export function SelectItem({ className, children, id, textValue }: SimpleProps & { id: string; textValue?: string }) {
  return <ListBoxItem id={id} textValue={textValue} className={mergeClassName('app-select-item', className)}>{({ isSelected }) => <>{children}<span className="app-select-check" aria-hidden="true">{isSelected && <Check size={15} strokeWidth={2.4} />}</span></>}</ListBoxItem>;
}
